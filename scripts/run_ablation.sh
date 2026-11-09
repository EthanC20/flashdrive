#!/usr/bin/env bash
# 在 tmux 里运行三条 clip 的 2x2 消融。每次只启动一个 Python 进程，跑完即退出，
# 避免模型权重和 CUDA graph 留在同一个进程里。
#
#   bash scripts/run_ablation.sh
#   NUM_STEPS=30 WARMUP_STEPS=3 bash scripts/run_ablation.sh

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -f "$ROOT/.venv/bin/activate" ]]; then
  echo "Missing $ROOT/.venv. Run 'uv sync' in the repo first."
  exit 1
fi

set +u
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
set -u

if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  echo "uv venv has no python interpreter at $ROOT/.venv/bin/python"
  exit 1
fi
if [[ "$(readlink -f "$(command -v python)")" != "$(readlink -f "$ROOT/.venv/bin/python")" ]]; then
  echo "Activated python is not the uv venv interpreter: $(command -v python)"
  exit 1
fi

NUM_STEPS="${NUM_STEPS:-120}"
WARMUP_STEPS="${WARMUP_STEPS:-3}"
if ! [[ "$NUM_STEPS" =~ ^[0-9]+$ && "$WARMUP_STEPS" =~ ^[0-9]+$ ]]; then
  echo "NUM_STEPS and WARMUP_STEPS must be non-negative integers."
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

if ! command -v flock >/dev/null 2>&1; then
  echo "flock is required so two tmux windows cannot load the model at once."
  exit 1
fi

mkdir -p "$ROOT/logs/ablation"
exec 9>"$ROOT/logs/ablation/.lock"
if ! flock -n 9; then
  echo "Another ablation run holds logs/ablation/.lock. Exit."
  exit 1
fi

STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="$ROOT/logs/ablation/$STAMP"
mkdir -p "$OUT"
MANIFEST="$OUT/manifest.tsv"
printf 'index\tclip_id\tconfig\tlog\tstart\tend\texit_code\tgpu_memory_mib\tcommand\n' >"$MANIFEST"

trap 'echo "Interrupted. Partial logs: $OUT"; exit 130' INT TERM

clips=(
  "87147a1b-3eef-4c25-94d2-ec7718a49a7a"
  "2ac89a4c-fa8e-4ad5-9aea-bab053f83ec1"
  "c84309d5-6f92-44f7-9170-a4ea6b13cc7b"
)
# name|extra infer.py flags. full 不加开关。
configs=(
  "base|--no-dflash --no-paroquant"
  "dflash|--no-paroquant"
  "w4a8|--no-dflash"
  "full|"
)

failures=0
index=0

echo "Ablation logs: $OUT"
echo "Runs: 12 sequential processes, num_steps=$NUM_STEPS warmup_steps=$WARMUP_STEPS"

for clip in "${clips[@]}"; do
  short="${clip%%-*}"
  for spec in "${configs[@]}"; do
    index=$((index + 1))
    name="${spec%%|*}"
    flags="${spec#*|}"
    printf -v idx "%02d" "$index"
    log="$OUT/${idx}_${short}_${name}.log"

    cmd=(
      python -u scripts/infer.py
      --model-path z-lab/Alpamayo-1.5-10B
      --stage-timing
      --num-steps "$NUM_STEPS"
      --warmup-steps "$WARMUP_STEPS"
      --num-traj-samples 1
      --clip-id "$clip"
    )
    if [[ -n "$flags" ]]; then
      # shellcheck disable=SC2206
      extra=($flags)
      cmd+=("${extra[@]}")
    fi

    start="$(date -Is)"
    echo "=== [${idx}/12] ${short} ${name} ==="
    {
      printf 'command:'
      printf ' %q' "${cmd[@]}"
      printf '\n'
      echo "start: ${start}"
      "${cmd[@]}"
    } 2>&1 | tee "$log"
    status="${PIPESTATUS[0]}"
    end="$(date -Is)"
    gpu_index="${CUDA_VISIBLE_DEVICES%%,*}"
    mem="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu_index" 2>/dev/null | head -n1 || true)"
    mem="${mem:-na}"
    {
      echo "end: ${end}"
      echo "exit: ${status}"
      echo "gpu_memory_mib: ${mem}"
    } | tee -a "$log"

    cmdline="$(printf '%q ' "${cmd[@]}")"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$idx" "$clip" "$name" "$(basename "$log")" "$start" "$end" "$status" "$mem" "$cmdline" \
      >>"$MANIFEST"

    if [[ "$status" -ne 0 ]]; then
      failures=$((failures + 1))
      echo "Run ${idx} failed with exit ${status}. Continuing."
    fi
  done
done

echo "Finished. Logs: $OUT"
echo "Manifest: $MANIFEST"
if [[ "$failures" -ne 0 ]]; then
  echo "${failures} run(s) failed."
  exit 1
fi
echo "All 12 runs finished."
