#!/usr/bin/env bash
# B200 / GB200 (SM100a) validation + TFLOPS sweep.
#
# Prereqs: CUDA driver 12.8+ and libnvrtc on the library path, e.g.
#   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
# or, without a toolkit install (pip wheel):
#   pip install nvidia-cuda-nvrtc-cu12
#   export LD_LIBRARY_PATH=$(python3 -c \
#     "import nvidia.cuda_nvrtc,os;print(os.path.dirname(nvidia.cuda_nvrtc.__file__)+'/lib')"):$LD_LIBRARY_PATH
#
# Usage:
#   ./scripts/b200_run.sh          # e2e correctness + full TFLOPS sweep
#   ./scripts/b200_run.sh e2e      # correctness only
#   ./scripts/b200_run.sh bench    # TFLOPS sweep only
#
# Every bench row is preceded by a `[deepgemm] ...: block=MxNxK stages=..
# cluster=.. swap_ab=..` line (DG_PRINT_CONFIGS), so TFLOPS can be
# correlated with the tile decision — paste config + TFLOPS pairs back
# into an issue to tune the heuristics for your exact shapes.
set -euo pipefail
cd "$(dirname "$0")/.."
MODE="${1:-all}"

case "$MODE" in
  e2e|all)
    echo "==== [1/2] end-to-end correctness (Blackwell paths) ===="
    cargo test -p deepgemm --features e2e --test e2e -- --nocapture
    ;;
esac

case "$MODE" in
  bench|all)
    echo "==== [2/2] TFLOPS sweep ===="
    B=(cargo run --release -p deepgemm-bench --)
    # --- MXFP4 (tcgen05.kind::mxf4, packed e2m1 + UE8M0 per-32) ---
    "${B[@]}" bench --op fp4_nt_native --m 8192 --n 8192 --k 7168
    "${B[@]}" bench --op fp4_nt_native --m 4096 --n 4096 --k 7168
    "${B[@]}" bench --op fp4_nt_native --m 4096 --n 7168 --k 7168
    "${B[@]}" bench --op fp4_nt_native --m 2048 --n 2048 --k 7168
    "${B[@]}" bench --op fp4_nt_native --m 512  --n 8192 --k 7168   # skinny M (swap-AB)
    "${B[@]}" bench --op fp4_nt_native --m 128  --n 8192 --k 7168   # very skinny M
    # --- MoE grouped MXFP4 ---
    "${B[@]}" bench --op fp4_grouped_contig  --groups 256 --m 128 --n 7168 --k 7168
    "${B[@]}" bench --op fp4_grouped_masked  --groups 128 --m 128 --n 2048 --k 7168
    # --- FP8 on SM100: DeepSeek recipe (gran 128) + MXFP8 (gran 32) ---
    "${B[@]}" bench --op fp8_nt_sm100  --m 4096 --n 7168 --k 7168
    "${B[@]}" bench --op mxfp8_nt_sm100 --m 4096 --n 7168 --k 7168
    # --- BF16 (tcgen05.kind::f16) ---
    "${B[@]}" bench --op bf16_nt_sm100 --m 8192 --n 8192 --k 8192
    ;;
esac
