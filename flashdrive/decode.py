"""关闭 DFlash 时的自回归解码。

采样和停止条件与 DFlash 对齐：temperature / top-p、屏蔽 trajectory token，
遇到 ``<cot_end>`` 或句号即停止，并在句号后补一个 ``<cot_end>`` 以便抽取 CoC。
每个写进返回序列的 token 都会占一个 KV 槽，trajectory-start 的 KV 用
``DFlashMixin._dflash_traj_forward`` 写入，这样动作专家的起点公式不用改。
"""

import logging
from typing import Any

import torch
from alpamayo1_5.models.token_utils import to_special_token

from flashdrive.dflash import _TrajectoryTokenMask, sample_tokens

logger = logging.getLogger(__name__)

# 句号后面补上的 cot_end，以及紧随其后的 trajectory-start。
# 不含投机块，只给这两个额外位置留出静态 cache。
AR_CACHE_SLACK = 2


class AutoregressiveMixin:
    """FlashDrive 自回归解码。不加载 draft，也不抓取 DFlash 的 hidden。"""

    def setup_autoregressive(self) -> None:
        """准备与 DFlash 相同的采样掩码和停止 token。"""
        self._use_dflash = False
        self._ar_traj_mask = _TrajectoryTokenMask(
            offset=self.config.traj_token_start_idx,
            size=self.config.traj_vocab_size,
        )
        self._ar_stop_token_id = self.tokenizer.convert_tokens_to_ids(to_special_token("cot_end"))
        # 与 DFlash 一样从 tokenizer 取句号，不用写死的 Qwen id。
        self._ar_period_token_id = self.tokenizer.encode(".")[0]
        logger.info("Autoregressive decode configured (DFlash off)")

    def _ar_prefill(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        cache_position: torch.Tensor,
        visual_pos_masks: torch.Tensor,
        deepstack_image_embeds: list[torch.Tensor],
        streaming_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Streaming prefill，只返回下一个 token 的 logits，不拼接 draft hidden。"""

        def prefill_fn(b: dict[str, Any]) -> torch.Tensor:
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
            return self.vlm.lm_head(output.last_hidden_state[:, -1])

        return self._compiled_step(
            "ar_prefill",
            {
                "inputs_embeds": inputs_embeds,
                "position_ids": position_ids,
                "cache_position": cache_position,
                "visual_pos_masks": visual_pos_masks,
                "deepstack_embeds": deepstack_image_embeds,
                "streaming_mask": streaming_attention_mask,
            },
            prefill_fn,
        )

    def _ar_token_step(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cache_position: torch.Tensor,
        temperature: float,
        top_p: float,
    ) -> torch.Tensor:
        """单 token 前向：写入当前 token 的 KV，并在图内采样下一个 token。

        ``temperature`` / ``top_p`` 在第一次调用时绑进图，之后的调用沿用该值。
        返回形状 ``[batch, 1]``。停止与否由调用方在图外读取。
        """

        def token_fn(b: dict[str, Any]) -> torch.Tensor:
            hidden = self.vlm.model.language_model(
                input_ids=b["input_ids"],
                position_ids=b["position_ids"],
                past_key_values=self._past_key_values,
                cache_position=b["cache_position"],
                use_cache=True,
            ).last_hidden_state
            logits = self.vlm.lm_head(hidden[:, -1])
            return sample_tokens(logits.unsqueeze(1), self._ar_traj_mask, temperature, top_p)

        return self._compiled_step(
            "ar_token",
            {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "cache_position": cache_position,
            },
            token_fn,
        )

    def _ar_decode(
        self,
        *,
        input_ids: torch.Tensor,
        logits: torch.Tensor,
        position_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> torch.Tensor:
        """从 prefill logits 开始逐 token 解码，直到停止，再写入 trajectory-start 的 KV。

        返回 ``[window tokens | CoT | traj_start]``，与 ``_dflash_generate`` 的布局一致。
        """
        if input_ids.shape[0] != 1:
            raise ValueError(
                f"Autoregressive decode supports batch size 1, got {input_ids.shape[0]}."
            )

        device = input_ids.device
        traj_start = torch.tensor([[self.traj_start_token_id]], device=device)
        if not self.has_compiled_step("dflash_traj_forward"):
            # 第一次编译 traj 前向时会重放另一张图，先把 prefill 的 logits 拷出来。
            logits = logits.clone()
            self._dflash_traj_forward(
                traj_start,
                position_ids,
                torch.tensor([self.prefill_seq_length], device=device, dtype=torch.long),
            )

        token = sample_tokens(logits.unsqueeze(1), self._ar_traj_mask, temperature, top_p)
        generated_ids: list[int] = []
        pos = self.prefill_seq_length
        while len(generated_ids) < max_new_tokens:
            token_id = int(token.item())
            generated_ids.append(token_id)
            cache_position = torch.tensor([pos], device=device, dtype=torch.long)
            is_stop = token_id == self._ar_stop_token_id
            is_period = token_id == self._ar_period_token_id
            if is_stop or is_period or len(generated_ids) == max_new_tokens:
                # 最后一个 CoT token 只需要 KV，不再采样。
                self._dflash_traj_forward(token, position_ids, cache_position)
                pos += 1
                if is_period and not is_stop:
                    generated_ids.append(self._ar_stop_token_id)
                    cot_end = torch.tensor([[self._ar_stop_token_id]], device=device)
                    self._dflash_traj_forward(
                        cot_end,
                        position_ids,
                        torch.tensor([pos], device=device, dtype=torch.long),
                    )
                    pos += 1
                break
            token = self._ar_token_step(token, position_ids, cache_position, temperature, top_p)
            pos += 1

        self._dflash_traj_forward(
            traj_start,
            position_ids,
            torch.tensor([pos], device=device, dtype=torch.long),
        )
        generated = torch.tensor(generated_ids, device=device, dtype=torch.long).view(1, -1)
        return torch.cat([input_ids, generated, traj_start], dim=-1)
