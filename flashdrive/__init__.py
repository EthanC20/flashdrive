"""Compose the FlashDrive feature patches onto a base Alpamayo model.

FlashDrive never edits upstream code; it patches at runtime, one mechanism per
level:

* the model instance: :func:`from_pretrained` synthesizes a subclass
  mixing the feature mixins ahead of the original class in the MRO;
* backbone submodules: ``patch_backbone`` swaps each Qwen3-VL module's
  ``__class__`` to a registered drop-in subclass (:mod:`flashdrive._backbone`);
* the Conv3D patch-embed: replaced module-globally at import time, the one
  patch the original-model path needs too (:mod:`flashdrive._backbone`).

Each mixin lives in its own module:

* :class:`flashdrive._base.FlashDriveBaseMixin` — shared scaffold
* :class:`flashdrive._compile.TorchCompileMixin` — torch.compile/cudagraph machinery
* :class:`flashdrive.streaming.StreamingMixin` — streaming
* :class:`flashdrive.dflash.DFlashMixin` — speculative decoding
* :class:`flashdrive.decode.AutoregressiveMixin` — autoregressive decode when DFlash is off
* :class:`flashdrive.quantization.ParoQuantMixin` — W4A8 quantization
* :class:`flashdrive.fusion.ExpertFusionMixin` — expert projection fusion
"""

import logging

import torch
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from transformers import PretrainedConfig

from flashdrive._base import FlashDriveBaseMixin
from flashdrive._compile import TorchCompileMixin
from flashdrive.decode import AutoregressiveMixin
from flashdrive.dflash import DFlashMixin
from flashdrive.fusion import ExpertFusionMixin
from flashdrive.quantization import ParoQuantMixin
from flashdrive.streaming import StreamingMixin

# Library convention: attach a no-op handler so importing FlashDrive never emits
# logs unless the application configures logging itself.
logger = logging.getLogger(__name__)
# Library convention: attach a no-op handler so importing FlashDrive never emits
# logs unless the application configures logging itself.
logger.addHandler(logging.NullHandler())


class FlashDriveMixin(
    FlashDriveBaseMixin,
    TorchCompileMixin,
    StreamingMixin,
    DFlashMixin,
    AutoregressiveMixin,
    ParoQuantMixin,
    ExpertFusionMixin,
):
    """All FlashDrive feature patches composed into a single mixin.

    Each mixin contributes one ``setup_*`` method; :func:`from_pretrained` applies
    them in dependency order (``setup_rollout`` first — it patches the backbone
    forwards the other setups build on). Mixins never call each other, and shared
    state has one writer: the base owns the cache and prompt token ids, streaming
    publishes the position tables, dflash publishes the block size. The
    autoregressive path is the exception: it reuses DFlash's one-token trajectory
    forward and does not load a draft.
    """


# The supported Alpamayo backbones, mapped to whether their streaming attention
# mask uses extended vision ranges: the Cosmos (Alpamayo 1.5) prompt places
# camera-name / frame-label text between vision blocks, so each view's KV range
# extends back over those tokens; Qwen3-VL (Alpamayo 1 / R1) packs frames
# back-to-back and uses tight per-frame ranges. Adding a backbone is one new row.
_EXTENDED_VISION_RANGES = {
    AlpamayoR1: False,
    Alpamayo1_5: True,
}


def resolve_model_class(model_path: str) -> type:
    """Resolve the Alpamayo model class from a checkpoint's ``model_type``."""
    model_type = PretrainedConfig.get_config_dict(model_path)[0]["model_type"]
    model_classes = {cls.config_class.model_type: cls for cls in _EXTENDED_VISION_RANGES}
    if model_type not in model_classes:
        raise ValueError(
            f"Unsupported model type {model_type!r}; supported: {sorted(model_classes)}."
        )
    return model_classes[model_type]


def _is_optimized(model_path: str) -> bool:
    """True for a z-lab (optimized) checkpoint; the originals live under nvidia."""
    return model_path.split("/", 1)[0].lower() == "z-lab"


def from_pretrained(
    model_path: str,
    device: str | torch.device = "cuda",
    torch_compile: str | None = "max-autotune",
    *,
    dflash: bool = True,
    paroquant: bool = True,
) -> torch.nn.Module:
    """Load an Alpamayo model from ``model_path`` — optimized FlashDrive or original.

    A ``z-lab`` checkpoint loads the FlashDrive stack. Streaming, action-cache
    hooks, expert fusion, and ``torch.compile`` always stay on. ``dflash`` loads
    the ``-DFlash`` draft; ``paroquant`` loads the ``-PARO`` W4A8 checkpoint.
    Either flag can be turned off for the ablation. An upstream ``nvidia``
    checkpoint loads the original model and ignores both flags. The Alpamayo
    release is resolved from the checkpoint's config and the model runs on sdpa
    (no flash-attn). ``torch_compile`` sets the compile mode for the rollout's
    primitives (``None`` = eager); it is ignored for the original model.
    """
    # Everything runs on sdpa, so nothing needs flash-attn. The alpamayo loader
    # builds the VLM from the custom ``attn_implementation``, transformers' load
    # gate reads the standard ``_attn_implementation``, and the wrapper class never
    # declared ``_supports_sdpa`` despite delegating to sdpa-capable submodels.
    model_cls = resolve_model_class(model_path)
    model_cls._supports_sdpa = True
    config = model_cls.config_class.from_pretrained(model_path)
    config.attn_implementation = config._attn_implementation = "sdpa"
    model = model_cls.from_pretrained(model_path, config=config, dtype=torch.bfloat16)
    model = model.to(device).eval()
    if not _is_optimized(model_path):
        if not dflash or not paroquant:
            logger.info(
                "Ignoring dflash=%s paroquant=%s for non-z-lab checkpoint %s",
                dflash,
                paroquant,
                model_path,
            )
        return model

    model.__class__ = type(f"FlashDrive{model_cls.__name__}", (FlashDriveMixin, model_cls), {})
    model.setup_rollout()
    model.setup_torch_compile(torch_compile)
    model.setup_streaming(extended_vision_ranges=_EXTENDED_VISION_RANGES[model_cls])
    if paroquant:
        model.setup_paroquant(f"{model_path}-PARO")
    model.setup_expert_fusion()
    if dflash:
        model.setup_dflash(f"{model_path}-DFlash")
    else:
        model.setup_autoregressive()
    logger.info(
        "FlashDrive features: streaming=on fusion=on compile=%s dflash=%s paroquant=%s",
        torch_compile,
        dflash,
        paroquant,
    )
    return model


__all__ = ["FlashDriveMixin", "from_pretrained", "resolve_model_class"]
