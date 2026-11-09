"""End-to-end inference: load a clip, run the model, report minADE + per-step latency.

The checkpoint org selects the mode: z-lab checkpoints run the optimized FlashDrive
stack (streaming + DFlash + W4A8 + expert fusion + action caching + compile), while
upstream (nvidia) checkpoints run the original model, for before/after timing. The
Alpamayo release (1 vs 1.5) is resolved from the checkpoint's config.

``--no-dflash`` and ``--no-paroquant`` turn those two features off on a z-lab
checkpoint. Streaming, action caching, expert fusion, and compile stay on.
``--stage-timing`` records Encode / Prefill / Decode / Action on that path.
"""

import argparse
import functools
import importlib
import inspect
import logging
import time
from collections.abc import Callable
from types import ModuleType
from typing import Any

import numpy as np
import physical_ai_av
import torch

# Importing flashdrive also applies, as a side effect, the torch 2.9 Conv3D->Linear
# fix (in flashdrive._backbone) that the original-model path relies on too.
import flashdrive
from flashdrive.streaming import convert_to_streaming_window

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Enable TF32 matmul for better perf on Ampere+ GPUs.
torch.set_float32_matmul_precision("high")

# Fixed optimized-path diffusion config.
MAX_NEW_TOKENS = 128
DIFFUSION_STEPS = 8
CACHE_STEPS = [3, 4, 5, 6]
DEFAULT_CLIP_ID = "87147a1b-3eef-4c25-94d2-ec7718a49a7a"
_STAGE_NAMES = ("encode", "prefill", "decode", "action")


def load_helper(model_path: str) -> ModuleType:
    """Import the ``helper`` module of the checkpoint's Alpamayo package
    (resolved from the checkpoint's config, not from the path)."""
    package = flashdrive.resolve_model_class(model_path).__module__.split(".")[0]
    return importlib.import_module(f"{package}.helper")


def create_inputs(
    helper: ModuleType,
    processor: Any,
    num_steps: int,
    clip_id: str,
    t0_us: int,
    time_step_us: int,
    streaming: bool,
) -> list[dict]:
    """Build per-window model inputs (up front, so data loading stays out of the timed loop).

    Optimized (streaming) runs keep window 0 as the full 16-frame prefill and convert
    windows 1+ to the 4-frame streaming form; the original model uses full windows.
    """
    package = helper.__name__.rsplit(".", 1)[0]
    load_dataset = importlib.import_module(
        f"{package}.load_physical_aiavdataset"
    ).load_physical_aiavdataset
    # Alpamayo 1.5 conditions its prompt on camera indices; detect from the helper.
    camera_conditioned = "camera_indices" in inspect.signature(helper.create_message).parameters

    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
    vision_start_token_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
    vision_end_token_id = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")

    ego_keys = ("ego_history_xyz", "ego_history_rot", "ego_future_xyz", "ego_future_rot")
    inputs_list = []
    for window_idx in range(num_steps):
        t0 = t0_us + window_idx * time_step_us
        data = load_dataset(clip_id, t0_us=t0, num_frames=4, avdi=avdi)
        frames = data["image_frames"].flatten(0, 1)  # (4, 4, C, H, W) -> (16, C, H, W)

        camera_kwargs = {"camera_indices": data["camera_indices"]} if camera_conditioned else {}
        messages = helper.create_message(frames, **camera_kwargs)

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )
        model_inputs = {"tokenized_data": inputs, **{k: data[k] for k in ego_keys}}
        if streaming and window_idx > 0:
            model_inputs = convert_to_streaming_window(
                model_inputs, vision_start_token_id, vision_end_token_id
            )
        inputs_list.append(model_inputs)

    return inputs_list


def compute_min_ade(gt_future_xyz: torch.Tensor, pred_xyz: torch.Tensor) -> tuple[float, int]:
    """Return (min ADE, sample index) of the predicted trajectories vs ground truth."""
    gt_xy = gt_future_xyz.cpu()[0, 0, :, :2].T.numpy()
    pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
    diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
    return float(diff.min()), int(diff.argmin())


def make_sample_fn(
    helper: ModuleType,
    model: torch.nn.Module,
    optimized: bool,
    device: str,
    num_traj_samples: int,
) -> Callable[[dict], tuple]:
    """Bind the per-run rollout once: mode, sampling kwargs, and device shipping.

    The returned callable maps ``model_inputs -> (pred_xyz, pred_rot, extra)``, so
    the benchmark loop stays mode-blind.
    """
    if optimized:
        rollout = functools.partial(
            model.sample_trajectories_streaming,
            max_new_tokens=MAX_NEW_TOKENS,
            diffusion_kwargs={
                "inference_step": DIFFUSION_STEPS,
                "cache_steps": CACHE_STEPS,
                "int_method": "euler_with_cache",
            },
        )
    else:
        rollout = model.sample_trajectories_from_data_with_vlm_rollout

    def sample(model_inputs: dict) -> tuple:
        data = helper.to_device(model_inputs, device)
        return rollout(data=data, num_traj_samples=num_traj_samples, return_extra=True)

    return sample


@torch.inference_mode()
def run_inference(sample: Callable, model_inputs: dict, log: bool = True) -> float:
    """Run one rollout; return its minADE."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, _, extra = sample(model_inputs)
    if pred_xyz is None:  # streaming first prefill only caches KV
        return float("inf")

    min_ade, min_ade_idx = compute_min_ade(model_inputs["ego_future_xyz"], pred_xyz)
    if log:
        logger.info("Chain-of-Causation:\n%s", extra["cot"][0][0][min_ade_idx])
        logger.info("MinADE: %.4f", min_ade)
    return min_ade


def _log_latency(latencies: list[float], *, label: str = "LATENCY_SUMMARY") -> None:
    logger.info(
        "%s ms: mean=%.1f median=%.1f min=%.1f p90=%.1f n=%d",
        label,
        np.mean(latencies),
        np.median(latencies),
        np.min(latencies),
        np.percentile(latencies, 90),
        len(latencies),
    )


def _log_stages(
    stages: list[dict[str, float]],
    latencies: list[float],
    generated: list[int],
    acceptance_sum: float,
    num_rounds: int,
    *,
    label: str = "STAGE_SUMMARY",
    decode_label: str = "DECODE_SUMMARY",
) -> None:
    """Log per-stage GPU time. ``latencies`` is aligned with ``stages`` (host e2e)."""
    if not stages:
        return
    columns = []
    sums = np.zeros(len(stages))
    for name in _STAGE_NAMES:
        values = np.array([row[name] for row in stages])
        sums += values
        columns.append(f"{name}_mean={values.mean():.1f} {name}_median={np.median(values):.1f}")
    e2e = np.asarray(latencies, dtype=np.float64)
    logger.info(
        "%s ms: %s sum_mean=%.1f e2e_mean=%.1f residual_mean=%.1f n=%d",
        label,
        " ".join(columns),
        float(sums.mean()),
        float(e2e.mean()),
        float((e2e - sums).mean()),
        len(stages),
    )
    acceptance = f"{acceptance_sum / num_rounds:.2f}" if num_rounds else "na"
    rounds = f"{num_rounds / len(generated):.2f}" if generated and num_rounds else "na"
    generated_mean = f"{float(np.mean(generated)):.2f}" if generated else "na"
    logger.info(
        "%s generated_mean=%s acceptance_mean=%s rounds_mean=%s n=%d",
        decode_label,
        generated_mean,
        acceptance,
        rounds,
        len(generated),
    )


def run_benchmark(
    sample: Callable,
    all_inputs: list[dict],
    warmup_steps: int,
    model: torch.nn.Module | None = None,
    stage_timing: bool = False,
) -> dict[str, Any]:
    """Warm up, then time per-window inference and report minADE + latency stats.

    When ``stage_timing`` is set, also read the four CUDA-event stages the
    FlashDrive rollout recorded. Window 0 has no stages and is left out of that
    summary. Returns the measured rows so several clips can be pooled.
    """
    logger.info("Warmup")
    for model_inputs in all_inputs[:warmup_steps]:
        run_inference(sample, model_inputs, log=False)
    logger.info("Warmup completed")

    min_ades: list[float] = []
    latencies: list[float] = []
    stage_rows: list[dict[str, float]] = []
    stage_latencies: list[float] = []
    generated: list[int] = []
    acceptance_sum = 0.0
    num_rounds = 0
    for step, model_inputs in enumerate(all_inputs[warmup_steps:], warmup_steps + 1):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        min_ades.append(run_inference(sample, model_inputs))
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000.0)
        stages = model.consume_stage_ms() if stage_timing and model is not None else None
        if stages is None:
            logger.info(
                "Step %d/%d completed | Latency: %.1f ms", step, len(all_inputs), latencies[-1]
            )
            continue
        stage_rows.append(stages)
        stage_latencies.append(latencies[-1])
        logger.info(
            "Step %d/%d completed | Latency: %.1f ms | encode=%.1f prefill=%.1f "
            "decode=%.1f action=%.1f",
            step,
            len(all_inputs),
            latencies[-1],
            stages["encode"],
            stages["prefill"],
            stages["decode"],
            stages["action"],
        )
        stats = getattr(model, "last_decode_stats", None)
        if stats:
            generated.append(int(stats["num_generated"]))
            acceptance_sum += float(stats.get("acceptance_sum", 0.0))
            num_rounds += int(stats.get("num_rounds", 0))

    logger.info("Average MinADE: %.4f", np.mean(min_ades))
    _log_latency(latencies)
    _log_stages(stage_rows, stage_latencies, generated, acceptance_sum, num_rounds)
    return {
        "min_ades": min_ades,
        "latencies": latencies,
        "stages": stage_rows,
        "stage_latencies": stage_latencies,
        "generated": generated,
        "acceptance_sum": acceptance_sum,
        "num_rounds": num_rounds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default="z-lab/Alpamayo-1.5-10B",
        help="Base checkpoint. z-lab runs optimized FlashDrive; nvidia runs the original model.",
    )
    parser.add_argument(
        "--no-dflash",
        action="store_true",
        help="On a z-lab checkpoint, decode autoregressively. Streaming, action cache, "
        "and system optimizations stay on.",
    )
    parser.add_argument(
        "--no-paroquant",
        action="store_true",
        help="On a z-lab checkpoint, keep the bf16 language model instead of W4A8.",
    )
    parser.add_argument(
        "--stage-timing",
        action="store_true",
        help="Record Encode, Prefill, Decode, and Action GPU time on the FlashDrive path.",
    )
    parser.add_argument("--num-steps", type=int, default=120, help="Total windows to run.")
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=3,
        help="Initial windows excluded from the metrics. Must be >= 1 on optimized runs: "
        "window 0 only prefills the streaming cache and returns no trajectories.",
    )
    parser.add_argument(
        "--num-traj-samples",
        type=int,
        default=1,
        help="Number of trajectory samples to draw per step.",
    )
    parser.add_argument(
        "--clip-id",
        action="append",
        dest="clip_ids",
        default=None,
        help=(
            f"PhysicalAI-AV clip to evaluate. Repeat for several clips. Default: {DEFAULT_CLIP_ID}."
        ),
    )
    parser.add_argument("--t0-us", type=int, default=1_700_000, help="Clip start time (us).")
    parser.add_argument("--time-step-us", type=int, default=100_000, help="Window stride (us).")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    helper = load_helper(args.model_path)
    model = flashdrive.from_pretrained(
        args.model_path,
        device=args.device,
        dflash=not args.no_dflash,
        paroquant=not args.no_paroquant,
    )
    optimized = isinstance(model, flashdrive.FlashDriveMixin)
    stage_timing = bool(args.stage_timing and optimized)
    if args.stage_timing and not optimized:
        logger.info(
            "Stage timing applies to the FlashDrive path only; reporting end-to-end latency."
        )
    if stage_timing:
        model.stage_timing = True
    processor = helper.get_processor(model.tokenizer)
    sample = make_sample_fn(helper, model, optimized, args.device, args.num_traj_samples)
    clip_ids = args.clip_ids or [DEFAULT_CLIP_ID]

    pooled: list[dict[str, Any]] = []
    for clip_id in clip_ids:
        if optimized:
            model.reset_stream()
        if len(clip_ids) > 1:
            logger.info("Clip %s", clip_id)
        logger.info("Loading inputs...")
        all_inputs = create_inputs(
            helper,
            processor,
            args.num_steps,
            clip_id,
            args.t0_us,
            args.time_step_us,
            streaming=optimized,
        )
        pooled.append(
            run_benchmark(
                sample,
                all_inputs,
                args.warmup_steps,
                model=model,
                stage_timing=stage_timing,
            )
        )

    if len(pooled) > 1:
        _log_latency(
            [value for run in pooled for value in run["latencies"]],
            label="POOLED_LATENCY_SUMMARY",
        )
        _log_stages(
            [row for run in pooled for row in run["stages"]],
            [value for run in pooled for value in run["stage_latencies"]],
            [value for run in pooled for value in run["generated"]],
            sum(run["acceptance_sum"] for run in pooled),
            sum(run["num_rounds"] for run in pooled),
            label="POOLED_STAGE_SUMMARY",
            decode_label="POOLED_DECODE_SUMMARY",
        )


if __name__ == "__main__":
    main()
