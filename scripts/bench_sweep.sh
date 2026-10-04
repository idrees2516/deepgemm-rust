#!/usr/bin/env bash
# DeepSeek-V3-style shape sweep.
set -euo pipefail
cd "$(dirname "$0")/.."
B="cargo run --release -p deepgemm-bench --"
$B bench --op fp8_nt --m 64    --n 7168 --k 7168   # decode
$B bench --op fp8_nt --m 128   --n 7168 --k 7168
$B bench --op fp8_nt --m 2048  --n 7168 --k 7168   # prefill
$B bench --op fp8_nt --m 4096  --n 4096 --k 7168
$B bench --op fp8_nt --m 8192  --n 2048 --k 7168
$B bench --op grouped --groups 256 --n 7168 --k 7168
$B bench --op grouped --groups 64  --n 2048 --k 7168
$B bench --op bf16_nt --m 8192 --n 8192 --k 8192
