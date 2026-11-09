"""Shared FlashDrive scaffold.

`FlashDriveBaseMixin` owns the rollout: `setup_rollout`, the compiled `_encode`
/ `_action` primitives, the StaticCache lifecycle, the shared action tail, and
the public `sample_trajectories_streaming` driver that orchestrates the feature
mixins. Primitive and phase names follow the paper's rollout stages — encode,
prefill, decode, action — and are pinned to them.
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import einops
import numpy as np
import torch
from alpamayo1_5.models.token_utils import (
    extract_text_tokens,
    replace_padding_after_eos,
    to_special_token,
)

from flashdrive._backbone import StaticCache, patch_backbone
from flashdrive.decode import AR_CACHE_SLACK
from flashdrive.diffusion import euler_with_cache

logger = logging.getLogger(__name__)

# (pred_xyz, pred_rot) with an extra text dict when ``return_extra``; every element
# is None on a stream's first (prefill-only) call.
_RolloutOutput = (
    tuple[torch.Tensor | None, torch.Tensor | None]
    | tuple[torch.Tensor | None, torch.Tensor | None, dict[str, np.ndarray] | None]
)


class FlashDriveBaseMixin:
    """Feature-agnostic FlashDrive machinery shared by all rollouts."""

    def setup_rollout(self) -> None:
        """Patch the backbone and initialize the rollout's shared state.

        Must run before the other setups: the fuse-aware forwards and the DFlash
        hidden-state capture live on the patched backbone classes.
        """
        # No cache = fresh stream; the first window allocates it and prefills.
        self._past_key_values = None
        # 默认走 DFlash。setup_autoregressive 会把它关掉。
        if not hasattr(self, "_use_dflash"):
            self._use_dflash = True
        self.stage_timing = False
        self._stage_events = None
        self.last_decode_stats = None
        self._acceptance_rounds = []

        self.traj_start_token_id = self.tokenizer.convert_tokens_to_ids(
            to_special_token("traj_future_start")
        )
        self.vision_start_token_id = self.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        self.vision_end_token_id = self.tokenizer.convert_tokens_to_ids("<|vision_end|>")

        patch_backbone(self)

    @contextmanager
    def _stage(self, name: str) -> Iterator[None]:
        """在 CUDA event 里包住一个阶段。计时关闭时不做任何事。"""
        if not self.stage_timing:
            yield
            return
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            if self._stage_events is None:
                self._stage_events = {}
            self._stage_events[name] = (start, end)

    def consume_stage_ms(self) -> dict[str, float] | None:
        """读取上一窗的阶段耗时（毫秒）。调用前需要先同步 CUDA。

        窗口 0 和未开启计时的运行返回 None。
        """
        events = self._stage_events
        self._stage_events = None
        if not events:
            return None
        return {name: start.elapsed_time(end) for name, (start, end) in events.items()}

    def reset_stream(self) -> None:
        """丢掉当前 clip 的 KV、streaming 状态和已编译图，下一次调用从窗口 0 开始。"""
        had_graphs = bool(getattr(self, "_compiled_step_registry", None))
        self._past_key_values = None
        self._frame_ranges = None
        self._traj_text_range = None
        self._shift_src_index = None
        self._shift_dst_index = None
        self.streaming_position_ids = None
        self.streaming_cache_position = None
        self._cached_streaming_attention_mask = None
        self._stage_events = None
        self.last_decode_stats = None
        self._acceptance_rounds = []
        if hasattr(self, "_action_noise"):
            del self._action_noise
        registry = getattr(self, "_compiled_step_registry", None)
        if registry is not None:
            registry.clear()
        visual_root = getattr(getattr(self, "vlm", None), "model", None)
        if visual_root is not None:
            if hasattr(visual_root.visual, "reset_shape_caches"):
                visual_root.visual.reset_shape_caches()
            if hasattr(visual_root.language_model, "reset_shape_caches"):
                visual_root.language_model.reset_shape_caches()
        if had_graphs:
            logger.info("Reset streaming state and compiled graphs for a new clip")

    # =========================== Properties ===========================

    @property
    def num_action_tokens(self) -> int:
        return self.action_space.get_action_space_dims()[0]

    # =========================== Utility methods ===========================

    def _find_traj_start_positions(self, output_ids: torch.Tensor) -> torch.Tensor:
        """Index of the first trajectory-start token per sequence (last index if absent)."""
        traj_start_mask = output_ids == self.traj_start_token_id
        has_traj_start = traj_start_mask.any(dim=1)

        if not has_traj_start.all():
            missing = (~has_traj_start).nonzero(as_tuple=True)[0].tolist()
            logger.warning("No <traj_future_start> token found in sequences: %s", missing)

        return torch.where(
            has_traj_start,
            traj_start_mask.int().argmax(dim=1),
            output_ids.shape[1] - 1,
        )

    def _embed_tokens_with_images(
        self, input_ids: torch.Tensor, image_embeds: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Embed tokens and scatter image features into the image-token slots.

        Returns ``(inputs_embeds, image_mask)``; ``image_mask`` is the per-element
        boolean mask the language-model forward reuses as ``visual_pos_masks``.
        """
        inputs_embeds = self.vlm.model.get_input_embeddings()(input_ids)
        image_mask = (
            (input_ids == self.vlm.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        return inputs_embeds, image_mask

    # =========================== Compiled primitives ===========================

    def _encode(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        def encode_fn(b: dict[str, Any]) -> tuple[torch.Tensor, list[torch.Tensor]]:
            pixels = b["pixel_values"].type(self.vlm.model.visual.dtype)
            return self.vlm.model.visual(pixels, grid_thw=b["image_grid_thw"])

        return self._compiled_step(
            "encode",
            {
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
            },
            encode_fn,
        )

    def _action(
        self,
        total_samples: int,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cache_position: torch.Tensor,
        diffusion_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        device = cache_position.device
        num_action_tokens = self.num_action_tokens
        # Noise is freshly sampled each call (not copied from an input), so it lives
        # outside _compiled_step's input buffers; the closure captures it once.
        if not hasattr(self, "_action_noise"):
            self._action_noise = torch.empty(
                total_samples,
                *self.action_space.get_action_space_dims(),
                device=device,
                dtype=torch.bfloat16,
            )
        action_noise = self._action_noise

        expert_kwargs = {"is_causal": False} if self.config.expert_non_causal_attention else {}
        action_dims = self.action_space.get_action_space_dims()
        sample_kwargs = dict(diffusion_kwargs or {})
        cached_euler = sample_kwargs.get("int_method") == "euler_with_cache"

        def action_fn(b: dict[str, Any]) -> torch.Tensor:
            def step_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
                action_embeds = self.action_in_proj(x, t)
                if action_embeds.dim() == 2:
                    action_embeds = action_embeds.view(x.shape[0], num_action_tokens, -1)

                hidden = self.expert(
                    inputs_embeds=action_embeds,
                    position_ids=b["position_ids"],
                    past_key_values=self._past_key_values,
                    attention_mask=b["attention_mask"],
                    cache_position=b["cache_position"],
                    use_cache=True,
                    **expert_kwargs,
                ).last_hidden_state[:, -num_action_tokens:]
                return self.action_out_proj(hidden).view(-1, *action_dims)

            if cached_euler:
                return euler_with_cache(
                    self.diffusion,
                    noise=action_noise,
                    batch_size=total_samples,
                    step_fn=step_fn,
                    device=device,
                    cache_steps=sample_kwargs["cache_steps"],
                    inference_step=sample_kwargs.get(
                        "inference_step", self.diffusion.num_inference_steps
                    ),
                )
            return self.diffusion.sample(
                noise=action_noise,
                batch_size=total_samples,
                step_fn=step_fn,
                device=device,
                return_all_steps=False,
                **sample_kwargs,
            )

        action_noise.normal_()

        return self._compiled_step(
            "action",
            {
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
            },
            action_fn,
        )

    # =========================== Shared action tail ===========================

    def _sample_actions_and_format(
        self,
        *,
        action_start_pos: torch.Tensor,
        num_traj_samples: int,
        num_traj_sets: int,
        ego_history_xyz: torch.Tensor,
        ego_history_rot: torch.Tensor,
        output_ids: torch.Tensor,
        diffusion_kwargs: dict[str, Any] | None,
        return_extra: bool,
    ) -> _RolloutOutput:
        """Action-expert diffusion sampling + trajectory formatting (shared tail)."""
        device = output_ids.device
        batch_size = ego_history_xyz.shape[0]
        num_samples = num_traj_samples * num_traj_sets
        action_start = action_start_pos[0].item()

        indices = torch.arange(self._past_key_values.max_cache_len, device=device).expand(
            num_samples, -1
        )
        is_prompt = indices < action_start_pos[:, None]
        is_action = (indices >= action_start) & (indices < action_start + self.num_action_tokens)
        attention_mask = torch.where(
            (is_prompt | is_action)[:, None, None, :], 0.0, torch.finfo(torch.float32).min
        )

        cache_position = torch.arange(
            action_start, action_start + self.num_action_tokens, device=device
        )

        sampled_action = self._action(
            total_samples=batch_size * num_samples,
            position_ids=self.streaming_position_ids,
            cache_position=cache_position,
            attention_mask=attention_mask,
            diffusion_kwargs=diffusion_kwargs,
        )

        hist_xyz = einops.repeat(ego_history_xyz[:, -1], "b ... -> (b n) ...", n=num_samples)
        hist_rot = einops.repeat(ego_history_rot[:, -1], "b ... -> (b n) ...", n=num_samples)
        pred_xyz, pred_rot = self.action_space.action_to_traj(sampled_action, hist_xyz, hist_rot)
        pred_xyz = einops.rearrange(
            pred_xyz, "(b ns nj) ... -> b ns nj ...", ns=num_traj_sets, nj=num_traj_samples
        )
        pred_rot = einops.rearrange(
            pred_rot, "(b ns nj) ... -> b ns nj ...", ns=num_traj_sets, nj=num_traj_samples
        )

        if return_extra:
            extra = extract_text_tokens(self.tokenizer, output_ids)
            for key in extra:
                extra[key] = np.broadcast_to(
                    np.array(extra[key]).reshape(batch_size, 1, 1),
                    [batch_size, num_traj_sets, num_traj_samples],
                ).copy()
            return pred_xyz, pred_rot, extra
        return pred_xyz, pred_rot

    # =========================== Entry point ===========================

    @torch.inference_mode()
    def sample_trajectories_streaming(
        self,
        data: dict[str, Any],
        max_new_tokens: int = 128,
        temperature: float = 0.6,
        top_p: float = 0.98,
        num_traj_samples: int = 6,
        num_traj_sets: int = 1,
        diffusion_kwargs: dict[str, Any] | None = None,
        return_extra: bool = False,
    ) -> _RolloutOutput:
        """Run one FlashDrive window: streaming prefill, decode, then the action expert.

        Decode is DFlash when ``_use_dflash`` is set, otherwise autoregressive.
        The first call per stream only prefills the KV cache and returns
        ``(None, None)``; it also fixes the stream's budget — ``max_new_tokens``
        and the sample counts size the static cache and cudagraph shapes, and
        ``temperature`` / ``top_p`` (defaults match the original model's sampled
        decoding; 0 = greedy) are bound into the decode graph — so later windows
        must not change them.
        """
        self._stage_events = None
        self.last_decode_stats = None
        tokenized = data["tokenized_data"]
        input_ids = tokenized["input_ids"]
        pixel_values = tokenized["pixel_values"]
        image_grid_thw = tokenized["image_grid_thw"]
        attention_mask = tokenized.get("attention_mask")
        ego_history_xyz = data["ego_history_xyz"]
        ego_history_rot = data["ego_history_rot"]

        batch_size, num_traj_groups, _, _ = ego_history_xyz.shape
        num_samples = num_traj_samples * num_traj_sets
        if num_traj_groups != 1:
            raise ValueError(f"Only one trajectory group is supported, got {num_traj_groups}.")

        input_ids = self.fuse_traj_tokens(
            input_ids, {"ego_history_xyz": ego_history_xyz, "ego_history_rot": ego_history_rot}
        )

        # ---- first window: size + allocate the stream state, prefill KV, bail out ----
        if self._past_key_values is None:
            self.prefill_seq_length = input_ids.shape[1]
            # DFlash 的 verify 会一次写下一块；自回归只为补写的 cot_end 和 traj token 留空。
            cache_slack = self.dflash_block_size if self._use_dflash else AR_CACHE_SLACK
            self.max_cache_len = (
                self.prefill_seq_length + max_new_tokens + cache_slack + self.num_action_tokens
            )
            self._past_key_values = StaticCache(
                config=self.vlm.config,
                max_cache_len=self.max_cache_len,
                max_batch_size=num_samples * batch_size,
            )
            self._first_prefill(input_ids, attention_mask, pixel_values, image_grid_thw)
            self._shift_streaming_kv_cache()
            return (None, None, None) if return_extra else (None, None)

        # ---- streaming prefill inputs (drop everything before the first frame) ----
        first_vision_start = torch.where(input_ids == self.vision_start_token_id)[1][0].item()
        input_ids = input_ids[:, first_vision_start:]
        window_seq_length = input_ids.shape[1]
        with self._stage("encode"):
            image_embeds, deepstack_image_embeds = self._encode(pixel_values, image_grid_thw)

        # Prefill 含图像特征散入和语言模型的 streaming forward。Decode 从第一个新 token 到 traj KV。
        with self._stage("prefill"):
            inputs_embeds, image_mask = self._embed_tokens_with_images(input_ids, image_embeds)
            streaming_attention_mask = self._ensure_streaming_attention_mask()
            if self._use_dflash:
                logits, target_hidden = self._dflash_prefill(
                    inputs_embeds=inputs_embeds,
                    position_ids=self.streaming_position_ids,
                    cache_position=self.streaming_cache_position,
                    visual_pos_masks=image_mask[..., 0],
                    deepstack_image_embeds=deepstack_image_embeds,
                    streaming_attention_mask=streaming_attention_mask,
                )
            else:
                logits = self._ar_prefill(
                    inputs_embeds=inputs_embeds,
                    position_ids=self.streaming_position_ids,
                    cache_position=self.streaming_cache_position,
                    visual_pos_masks=image_mask[..., 0],
                    deepstack_image_embeds=deepstack_image_embeds,
                    streaming_attention_mask=streaming_attention_mask,
                )

        with self._stage("decode"):
            if self._use_dflash:
                output_ids = self._dflash_decode(
                    input_ids=input_ids,
                    logits=logits,
                    target_hidden=target_hidden,
                    position_ids=self.streaming_position_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
            else:
                output_ids = self._ar_decode(
                    input_ids=input_ids,
                    logits=logits,
                    position_ids=self.streaming_position_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )

        if self.stage_timing:
            # 返回序列末尾多一个 trajectory-start，不计入生成长度。
            stats: dict[str, Any] = {
                "num_generated": int(output_ids.shape[1] - window_seq_length - 1)
            }
            if self._use_dflash:
                rounds = self._acceptance_rounds
                stats["acceptance_sum"] = float(sum(rounds))
                stats["num_rounds"] = len(rounds)
            self.last_decode_stats = stats

        # ---- shared action-expert tail ----
        output_ids = replace_padding_after_eos(
            token_ids=output_ids,
            eos_token_id=self.traj_start_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        traj_start_pos = self._find_traj_start_positions(output_ids)

        if num_samples > 1:
            self._past_key_values.expand_batch()

        with self._stage("action"):
            result = self._sample_actions_and_format(
                action_start_pos=self.prefill_seq_length + (traj_start_pos - window_seq_length) + 1,
                num_traj_samples=num_traj_samples,
                num_traj_sets=num_traj_sets,
                ego_history_xyz=ego_history_xyz,
                ego_history_rot=ego_history_rot,
                output_ids=output_ids,
                diffusion_kwargs=diffusion_kwargs,
                return_extra=return_extra,
            )

        self._shift_streaming_kv_cache()
        return result
