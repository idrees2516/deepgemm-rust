#!/usr/bin/env bash
# Full GPU test + verify run (needs a CUDA driver + Hopper GPU for FP8).
set -euo pipefail
cd "$(dirname "$0")/.."
cargo test --workspace
cargo test -p deepgemm --features e2e --test e2e -- --nocapture
cargo run --release -p deepgemm-bench -- verify --op fp8_nt
cargo run --release -p deepgemm-bench -- verify --op grouped --groups 8 --n 256 --k 512
