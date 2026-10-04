#!/usr/bin/env bash
# Full GPU test + verify run (Hopper or Blackwell; arch-mismatched tests skip).
# FP8 kernels need Hopper (SM90a); FP4/MXFP8 kernels need Blackwell (SM100a).
set -euo pipefail
cd "$(dirname "$0")/.."
cargo test --workspace
cargo test -p deepgemm --features e2e --test e2e -- --nocapture
cargo run --release -p deepgemm-bench -- verify --op fp8_nt
cargo run --release -p deepgemm-bench -- verify --op grouped --groups 8 --n 256 --k 512
