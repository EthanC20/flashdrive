"""DFlash speculative-decoding patch for FlashDrive.

`DFlashMixin` isolates draft-model speculative decoding: draft model setup, the
compiled prefill/block-step/traj-forward primitives, the speculative decode loop,
and the `_dflash_generate` phase that `sample_trajectories_streaming`
dispatches to. This module also defines the small Qwen3-based draft network
(`DFlashDraftModel`). Streaming-specific inputs (position table, attention
mask) are passed in by the driver.
"""

import logging
from pathlib import Path
from typing import Any, NamedTuple

import torch
from alpamayo1_5.models.token_utils import to_special_token
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    Qwen3Config,
    Qwen3MLP,
    Qwen3PreTrainedModel,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    eager_attention_forward,
    rotate_half,
)

logger = logging.getLogger(__name__)


class _TrajectoryTokenMask(NamedTuple):
    """Trajectory-token span that `sample_tokens` masks out (the CoT is text-only)."""

    offset: int
    size: int


def sample_tokens(
    logits: torch.Tensor,
    traj_mask: _TrajectoryTokenMask,
    temperature: float,
    top_p: float = 1.0,
) -> torch.Tensor:
    """Decode tokens with the trajectory-token span masked to -inf (in place).

    Greedy at ``temperature=0``; otherwise temperature-scaled nucleus sampling,
    matching the original model's decoding (transformers' ``TopPLogitsWarper``).
    ``logits`` is mutated by the masking; callers pass fresh graph outputs.
    """
    offset, size = traj_mask
    logits[:, :, offset : offset + size] = float("-inf")
    if temperature <= 0:
        return torch.argmax(logits, dim=-1)
    batch_size, seq_len, vocab_size = logits.shape
    probs = torch.softmax(logits.view(-1, vocab_size) / temperature, dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
        # Drop tokens outside the nucleus: those whose preceding cumulative mass
        # already exceeds top_p (the top-1 token is always kept).
        outside = (sorted_probs.cumsum(dim=-1) - sorted_probs) > top_p
        sorted_probs[outside] = 0.0
        probs = torch.zeros_like(probs).scatter_(-1, sorted_idx, sorted_probs)
    return torch.multinomial(probs, num_samples=1).view(batch_size, seq_len)


def apply_rotary_pos_emb_tail(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """RoPE where q covers only the tail positions of the (context + noise) k span."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_len = q.size(-2)
    q_embed = (q * cos[..., -q_len:, :]) + (rotate_half(q) * sin[..., -q_len:, :])
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen3DFlashAttention(nn.Module):
    """Draft-model attention: the noise block attends to itself + the target context.

    Vendored from HF's Qwen3 attention; the context keys are prepended to the
    noise block, and the draft is stateless (no KV cache) and inference-only.
    """

    def __init__(self, config: Qwen3Config, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.is_causal = False
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.sliding_window = (
            config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        **kwargs: Any,
    ) -> torch.Tensor:
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden.shape[1]
        q = self.q_proj(hidden_states)
        q = q.view(bsz, q_len, -1, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)
        k_ctx = self.k_proj(target_hidden)
        k_noise = self.k_proj(hidden_states)
        v_ctx = self.v_proj(target_hidden)
        v_noise = self.v_proj(hidden_states)
        k = torch.cat([k_ctx, k_noise], dim=1).view(bsz, ctx_len + q_len, -1, self.head_dim)
        v = torch.cat([v_ctx, v_noise], dim=1).view(bsz, ctx_len + q_len, -1, self.head_dim)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_tail(q, k, cos, sin)
        attn_fn = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attn_fn = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attn_output, _ = attn_fn(
            self,
            q,
            k,
            v,
            attention_mask,
            dropout=0.0,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        return self.o_proj(attn_output.reshape(bsz, q_len, -1))


class Qwen3DFlashDecoderLayer(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3DFlashAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        **kwargs: Any,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn(
            hidden_states=self.input_layernorm(hidden_states),
            target_hidden=target_hidden,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        return residual + hidden_states


class DFlashDraftModel(Qwen3PreTrainedModel):
    config_class = Qwen3Config
    _no_split_modules = ["Qwen3DFlashDecoderLayer"]

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__(config)
        self.config = config
        self.layers = nn.ModuleList(
            [
                Qwen3DFlashDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.target_layer_ids = config.target_layer_ids
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config)
        self.fc = nn.Linear(
            len(self.target_layer_ids) * config.hidden_size, config.hidden_size, bias=False
        )
        self.hidden_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.block_size = getattr(config, "block_size", 8)
        self.context_len = getattr(config, "context_len", 1)
        self.post_init()

    def forward(
        self,
        position_ids: torch.LongTensor,
        noise_embedding: torch.Tensor,
        target_hidden: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        hidden_states = noise_embedding
        target_hidden = self.hidden_norm(self.fc(target_hidden))
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden=target_hidden,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        return self.norm(hidden_states)


class DFlashMixin:
    """FlashDrive draft-model speculative decoding behaviour."""

    # =========================== Setup ===========================

    def setup_dflash(self, checkpoint_path: str) -> None:
        """Load a DFlash draft checkpoint and wire speculative decoding onto the model.

        The draft's config must define ``mask_token_id``; block size, context length,
        and the target capture layers all come from the checkpoint.
        """
        self._use_dflash = True
        param = next(self.parameters())

        logger.info("Loading DFlash draft model from %s", checkpoint_path)
        draft_model = (
            DFlashDraftModel.from_pretrained(checkpoint_path, dtype=param.dtype)
            .to(param.device)
            .eval()
        )
        mask_token_id = getattr(draft_model.config, "mask_token_id", None)
        if mask_token_id is None:
            raise ValueError(f"{checkpoint_path!r} config defines no mask_token_id.")

        # The mask token sits one past the target's vocabulary; grow the embedding table.
        # No mean-resizing: the new embedding row is overwritten from the checkpoint
        # below, and mask tokens are filtered from the decoded output regardless.
        vocab_size = self.vlm.get_input_embeddings().weight.shape[0]
        if mask_token_id == vocab_size:
            self.vlm.resize_token_embeddings(vocab_size + 1, mean_resizing=False)
        elif mask_token_id > vocab_size:
            raise ValueError(
                f"mask_token_id={mask_token_id} is out of vocabulary range ({vocab_size})."
            )

        # The draft checkpoint ships the exact mask-token embedding it was trained with;
        # load it into the target's embedding table (from a local dir or the hub).
        mask_emb_file = Path(checkpoint_path) / "mask_embedding.pt"
        if not mask_emb_file.exists():
            from huggingface_hub import hf_hub_download

            mask_emb_file = Path(hf_hub_download(checkpoint_path, "mask_embedding.pt"))
        logger.debug("Loading mask embedding from %s", mask_emb_file)
        mask_emb = torch.load(mask_emb_file, map_location=param.device, weights_only=True)
        embeddings = self.vlm.get_input_embeddings()
        with torch.no_grad():
            embeddings.weight[mask_token_id] = mask_emb.to(dtype=embeddings.weight.dtype)

        self._draft_model = draft_model
        self.dflash_block_size = draft_model.block_size
        self._dflash_context_len = draft_model.context_len
        self._dflash_mask_token_id = mask_token_id
        self._dflash_traj_mask = _TrajectoryTokenMask(
            offset=self.config.traj_token_start_idx,
            size=self.config.traj_vocab_size,
        )
        self._dflash_stop_token_id = self.tokenizer.convert_tokens_to_ids(
            to_special_token("cot_end")
        )
        # Tokenizer-resolved: "."'s id is vocabulary-specific (13 is Qwen-only).
        self._dflash_period_token_id = self.tokenizer.encode(".")[0]

        # The (patched) text model captures the draft's context layers during forward.
        self.vlm.model.language_model.set_capture_layer_ids(draft_model.target_layer_ids)

        logger.info(
            "DFlash configured: block_size=%d, context_len=%s, target_layers=%s, mask_token_id=%d",
            self.dflash_block_size,
            self._dflash_context_len,
            draft_model.target_layer_ids,
            self._dflash_mask_token_id,
        )

    # =========================== Compiled primitives ===========================

    def _dflash_prefill(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        cache_position: torch.Tensor,
        visual_pos_masks: torch.Tensor,
        deepstack_image_embeds: list[torch.Tensor],
        streaming_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        def dflash_prefill_fn(b: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
            output = self.vlm.model.language_model(
                inputs_embeds=b["inputs_embeds"],
                position_ids=b["position_ids"],
                past_key_values=self._past_key_values,
                cache_position=b["cache_position"],
                visual_pos_masks=b["visual_pos_masks"],
                deepstack_visual_embeds=b["deepstack_embeds"],
                streaming_attention_mask=b["streaming_mask"],
                use_cache=True,
            )
            logits = self.vlm.lm_head(output.last_hidden_state[:, -1])
            context = torch.cat(output.hidden_states, dim=-1)[:, -self._dflash_context_len :, :]
            return logits, context

        return self._compiled_step(
            "dflash_prefill",
            {
                "inputs_embeds": inputs_embeds,
                "position_ids": position_ids,
                "cache_position": cache_position,
                "visual_pos_masks": visual_pos_masks,
                "deepstack_embeds": deepstack_image_embeds,
                "streaming_mask": streaming_attention_mask,
            },
            dflash_prefill_fn,
        )

    def _dflash_traj_forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cache_position: torch.Tensor,
    ) -> None:
        """Run the 1-token trajectory-start step through the target, for its KV only.

        Fires only when the verify missed the traj-KV piggyback; the driver warms
        it in the first window so its compile never lands mid-stream.
        """

        def traj_forward_fn(b: dict[str, Any]) -> torch.Tensor:
            return self.vlm.model.language_model(
                input_ids=b["input_ids"],
                position_ids=b["position_ids"],
                past_key_values=self._past_key_values,
                cache_position=b["cache_position"],
                use_cache=True,
            ).last_hidden_state

        self._compiled_step(
            "dflash_traj_forward",
            {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "cache_position": cache_position,
            },
            traj_forward_fn,
        )

    def _dflash_block_step(
        self,
        block_output_ids: torch.Tensor,
        target_hidden: torch.Tensor,
        position_ids: torch.Tensor,
        cache_position: torch.Tensor,
        temperature: float,
        top_p: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One speculative round in a single graph: draft, verify, sample, decide.

        Returns ``(block, posterior, context, stats)`` — the verified token block,
        the posterior samples, the draft context hidden states, and ``stats``
        packing ``[acceptance_length, first_stop_position_or_-1, period_hit]`` —
        so the decode loop pays one graph replay and one device sync per round.
        ``temperature`` / ``top_p`` are bound into the graph on first use.

        The draft proposes greedily: emitted tokens are always the posterior's,
        so the draft's decoding only affects the acceptance rate, and greedy
        maximises it.
        """

        def block_step_fn(
            b: dict[str, Any],
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            # Draft + greedy proposals.
            noise = self.vlm.model.language_model.embed_tokens(b["block_ids"])
            draft_positions = (
                torch.arange(self._dflash_context_len + self.dflash_block_size, device=noise.device)
                .unsqueeze(0)
                .expand(b["block_ids"].shape[0], -1)
            )
            hidden = self._draft_model(
                target_hidden=b["target_hidden"],
                noise_embedding=noise,
                position_ids=draft_positions,
                is_causal=False,
            )
            draft_logits = self.vlm.lm_head(hidden[:, 1:, :])
            proposed = sample_tokens(draft_logits, self._dflash_traj_mask, 0.0)
            # After a proposed <cot_end>, propose the trajectory-start token: if
            # that stop is accepted, this verify has already written the traj
            # token's KV (right token, position, and accepted-prefix context),
            # so the separate one-token traj forward can be skipped.
            follows_stop = nn.functional.pad(proposed[:, :-1] == self._dflash_stop_token_id, (1, 0))
            proposed = torch.where(
                follows_stop, proposed.new_tensor(self.traj_start_token_id), proposed
            )
            block = torch.cat([b["block_ids"][:, :1], proposed], dim=1)

            # Verify through the target (writes KV at cache_position).
            output = self.vlm.model.language_model(
                input_ids=block,
                position_ids=b["position_ids"],
                past_key_values=self._past_key_values,
                cache_position=b["cache_position"],
                use_cache=True,
            )
            verify_logits = self.vlm.lm_head(output.last_hidden_state)
            context = torch.cat(output.hidden_states, dim=-1)
            posterior = sample_tokens(verify_logits, self._dflash_traj_mask, temperature, top_p)

            # Round decisions. The emitted tokens are block[0..acceptance] plus
            # the bonus token at slot acceptance + 1. gather/scatter keep the
            # data-dependent index as a tensor — plain indexing would read it
            # to a Python scalar, a fullgraph break.
            matches = block[:, 1:] == posterior[:, :-1]
            acceptance = matches.cumprod(dim=1).sum(dim=1)[0]
            slot = acceptance.unsqueeze(0)
            bonus = posterior[0].gather(0, slot)
            emission = torch.cat([block[0], bonus]).scatter(0, slot + 1, bonus)
            slots = torch.arange(emission.shape[0], device=emission.device)
            stop_hits = (emission == self._dflash_stop_token_id) & (slots <= acceptance + 1)
            first_stop = torch.where(
                stop_hits.any(), stop_hits.int().argmax(), acceptance.new_tensor(-1)
            )
            period_hit = (bonus[0] == self._dflash_period_token_id).long()
            stats = torch.stack([acceptance, first_stop, period_hit])
            return block, posterior, context, stats

        return self._compiled_step(
            "dflash_block_step",
            {
                "block_ids": block_output_ids,
                "target_hidden": target_hidden,
                "position_ids": position_ids,
                "cache_position": cache_position,
            },
            block_step_fn,
        )

    # =========================== Speculative decode loop ===========================

    def _dflash_decode_loop(
        self,
        output_ids: torch.Tensor,
        num_input_tokens: int,
        position_ids: torch.Tensor,
        target_hidden: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> tuple[torch.Tensor, int, bool]:
        """Draft/verify blocks until ``<cot_end>``, a trailing period, or the budget.

        Also returns whether the final verify already cached the trajectory-start
        token's KV (see ``_dflash_block_step``), so the caller can skip the
        one-token traj forward.
        """
        device = output_ids.device
        block_size = self.dflash_block_size
        context_len = self._dflash_context_len
        stop_token_id = self._dflash_stop_token_id
        max_length = num_input_tokens + max_new_tokens

        start = num_input_tokens  # First known token is at output_ids[:, num_input_tokens]
        current_seq_len = num_input_tokens
        traj_cached = False
        target_hidden = target_hidden.clone()  # Detach from CUDA graph output

        while start < max_length:
            verify_cache_position = torch.arange(
                current_seq_len, current_seq_len + block_size, device=device
            )
            block, posterior, verify_context, stats = self._dflash_block_step(
                output_ids[:, start : start + block_size],
                target_hidden,
                position_ids,
                verify_cache_position,
                temperature,
                top_p,
            )
            acceptance_length, stop_at, period_flag = stats.tolist()
            if self.stage_timing:
                self._acceptance_rounds.append(int(acceptance_length))

            output_ids[:, start : start + acceptance_length + 1] = block[:, : acceptance_length + 1]
            output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]

            stop_position = None
            tokens_to_advance = acceptance_length + 1

            if stop_at >= 0:
                stop_position = stop_at
                tokens_to_advance = stop_position
                if stop_position < acceptance_length + 1:
                    output_ids[:, start + stop_position + 1 :] = self._dflash_mask_token_id
            elif period_flag:
                stop_position = acceptance_length + 1
                # A trailing period ends the CoT without an explicit <cot_end>;
                # write one so the CoC extraction sees a complete span.
                output_ids[:, start + acceptance_length + 2] = stop_token_id

            start += tokens_to_advance

            if stop_position is not None:
                # Piggybacked traj KV: the stop was an accepted in-block token with
                # a forced trajectory-start slot after it.
                traj_cached = 1 <= stop_at <= acceptance_length and stop_at + 1 < block_size
                current_seq_len += min(stop_position + 1, block_size)
                break

            # Rejected-draft KV needs no cleanup: the next block's verify overwrites
            # exactly those slots before any query can attend them (A/B-validated).
            current_seq_len += acceptance_length + 1

            # Slide the draft's context window over the newly accepted hidden states
            # (the cat copies them out of the verify graph's reusable output buffer).
            target_hidden = torch.cat(
                [target_hidden, verify_context[:, : acceptance_length + 1]], dim=1
            )[:, -context_len:]

        output_ids = output_ids[:, :max_length]
        mask = output_ids[0] != self._dflash_mask_token_id
        output_ids = output_ids[:, mask]

        # Truncate everything after the first stop token in the generated span.
        stop_indices = (output_ids[0, num_input_tokens:] == stop_token_id).nonzero(as_tuple=True)[0]
        if stop_indices.numel() > 0:
            output_ids = output_ids[:, : num_input_tokens + stop_indices[0] + 1]

        return output_ids, current_seq_len, traj_cached

    # =========================== Generate phase ===========================

    def _dflash_generate(
        self,
        *,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        visual_pos_masks: torch.Tensor,
        deepstack_image_embeds: list[torch.Tensor],
        position_ids: torch.Tensor,
        streaming_attention_mask: torch.Tensor,
        cache_position: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> torch.Tensor:
        """Streaming prefill + speculative decode until ``<cot_end>``.

        Returns the full token sequence (window input + generated CoT + the
        trajectory-start token); the shared action-expert tail runs in the driver.
        """
        logits, target_hidden = self._dflash_prefill(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_image_embeds=deepstack_image_embeds,
            streaming_attention_mask=streaming_attention_mask,
        )
        return self._dflash_decode(
            input_ids=input_ids,
            logits=logits,
            target_hidden=target_hidden,
            position_ids=position_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    def _dflash_decode(
        self,
        *,
        input_ids: torch.Tensor,
        logits: torch.Tensor,
        target_hidden: torch.Tensor,
        position_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> torch.Tensor:
        """Speculative decode from prefill logits until ``<cot_end>``, then the traj token.

        Returns the same layout as :meth:`_dflash_generate`: window tokens, CoT, and
        one trajectory-start token.
        """
        device = input_ids.device
        batch_size = input_ids.shape[0]
        if self.stage_timing:
            self._acceptance_rounds = []

        num_input_tokens = self.prefill_seq_length
        max_length = num_input_tokens + max_new_tokens
        dflash_output_ids = torch.full(
            (batch_size, max_length + self.dflash_block_size),
            self._dflash_mask_token_id,
            dtype=torch.long,
            device=device,
        )
        dflash_output_ids[:, :num_input_tokens] = self.tokenizer.pad_token_id

        traj_start_token = torch.tensor([[self.traj_start_token_id]], device=device)
        if not self.has_compiled_step("dflash_traj_forward"):
            # Compile the conditionally-fired traj forward with the rest of this
            # first window's graphs; its first fire must never compile mid-stream.
            # Writes the first decode slot, which the verify below overwrites. Clone
            # the prefill's live graph outputs first: running another graph would
            # invalidate them.
            logits, target_hidden = logits.clone(), target_hidden.clone()
            self._dflash_traj_forward(
                traj_start_token, position_ids, torch.tensor([num_input_tokens], device=device)
            )

        first_token = sample_tokens(logits.unsqueeze(1), self._dflash_traj_mask, temperature, top_p)
        dflash_output_ids[:, num_input_tokens : num_input_tokens + 1] = first_token

        dflash_output_ids, current_seq_len, traj_cached = self._dflash_decode_loop(
            output_ids=dflash_output_ids,
            num_input_tokens=num_input_tokens,
            position_ids=position_ids,
            target_hidden=target_hidden,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        if not traj_cached:
            # Feed the trajectory-start token through the target so its KV is cached
            # before the action expert runs.
            traj_cache_position = torch.tensor([current_seq_len], device=device, dtype=torch.long)
            self._dflash_traj_forward(traj_start_token, position_ids, traj_cache_position)

        generated_tokens = dflash_output_ids[:, num_input_tokens:]
        return torch.cat([input_ids, generated_tokens, traj_start_token], dim=-1)
