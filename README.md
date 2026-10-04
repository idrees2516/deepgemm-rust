# deepgemm-rust

**DeepGEMM in pure Rust** — fine-grained-scaled FP8 GEMM and MoE grouped GEMM for NVIDIA
GPUs, ported from [DeepGEMM-Ascend](https://github.com/deepseek-ai/DeepGEMM-Ascend) /
[DeepGEMM](https://github.com/deepseek-ai/DeepGEMM).

No PyTorch. No nvcc. No build step. Kernels are **CUDA C++ JIT-compiled at runtime with
NVRTC** (exactly like upstream DeepGEMM) and cached on disk — but the entire host stack,
the kernel launchers, the TMA descriptor builders, the heuristics, the scale-factor
transforms, the reference implementations and the benchmarks are **100% Rust**
([cudarc](https://crates.io/crates/cudarc) for the driver/NVRTC/cuBLASLt FFI).

```
Host (Rust)                                    Device (JIT-compiled sm_90a CUDA)
┌──────────────────────────────────┐           ┌─────────────────────────────────────┐
│ ops::fp8_gemm_nt / grouped_*     │  TMA maps │  warp-specialized persistent kernel │
│  ├─ heuristics (block config)    │ ────────► │   ├─ 1 producer warp: TMA loads     │
│  ├─ transform_sf (MN-major)      │  by-value │   ├─ 2 consumer warp groups: WGMMA  │
│  ├─ jit.rs (NVRTC + disk cache)  │  params   │   ├─ mbarrier pipeline (N stages)  │
│  └─ cuLaunchKernelEx (+cluster)  │           │   └─ STSM + TMA-store epilogue     │
└──────────────────────────────────┘           └─────────────────────────────────────┘
```


## Blackwell-native FP4/FP8 (SM100a, tcgen05)

On B200/GB200-class GPUs the crate dispatches to **5th-generation tensor
core** kernels, a 1:1 port of upstream DeepGEMM's SM100 family — written as
self-contained CUDA C++ with raw PTX (`tcgen05.*`), JIT-compiled with NVRTC
at runtime for `sm_100a`:

| kernel | MMA | notes |
|---|---|---|
| `sm100_fp8_fp4_gemm_1d1d` | `tcgen05.mma.kind::mxf4` (FP4xFP4, packed, UMMA_K=64, `scale_vec::2X`/`block32`) and `kind::mxf8f6f4` (FP8xFP8 / FP8xFP4, UMMA_K=32) | all 7 GemmTypes, swap-AB, 2-CTA `cta_group::2` multicast |
| `sm100_bf16_gemm` | `tcgen05.mma.kind::f16` | stage-merge optimization + tensor-core utilization control |

Key techniques (all ported, none stubbed):

* **Tensor-memory (TMEM) accumulation** with double-buffered accumulator
  stages and the upstream **SF/accumulator column-overlap trick** that packs
  scale factors next to live accumulators to fit the 512-column budget.
* **Scale factors ride the tensor cores**: UE8M0 (power-of-two) scales are
  TMA-loaded, in-warp transposed to the UTCCP layout, copied SMEM->TMEM with
  `tcgen05.cp.32x128b.warpx4`, and consumed directly by the block-scaled MMA
  (`a_sf_id`/`b_sf_id` select the packed byte) — zero FMA work spent on
  scaling, unlike the Hopper path.
* **Warp specialization**: TMA-producer warp, MMA-issue warp (leader CTA),
  UTCCP-transposer warps, epilogue warpgroup(s) with `tcgen05.ld` ->
  swizzled STSM -> TMA store (or `cp.reduce.async.bulk` for accumulation).
* **Epilogues**: identity, `alpha`, stochastic-round-to-BF16 (`cvt.rs`), and
  **quantize-to-FP8 with dynamic per-row per-32 UE8M0 output SFs** (batched).
* **Full recipes**: gran-32 (MXFP4/MXFP8) and gran-128 (DeepSeek) on either
  side, K-major and MN-major operands (FP4 is K-major only, like upstream),
  and every layout `nt/nn/tt/tn` natively.
* **Blackwell-native quantization**: `cvt.rn.satfinite.e2m1x2.f32` (the
  fused F2FP+PACK idiom) for FP4, `e4m3x2` for FP8; packed-FP4 TMA via the
  `CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B/16B` descriptors.
* **Grouped SF layout parity**: for masked/batched/contiguous grouped GEMMs
  the scale factors use upstream's `(tma_aligned(group_rows), cols * groups)`
  layout with the group index folded into the SF *column* coordinate (the
  kernel reads `group * shape_sf_k + col`), for both the A and B sides.

### The `mk_alignment` knob (MoE tile width)

Upstream's `set_mk_alignment_for_contiguous_layout` controls both the tile
shape the heuristics pick for m-grouped GEMMs and the layout contract the
caller must honor (each group's tokens start at an aligned row of the A
arena). This crate mirrors it exactly:

```rust
use deepgemm::types::{
    get_theoretical_mk_alignment_for_contiguous_layout,
    set_mk_alignment_for_contiguous_layout,
};

// SM100's UMMA_N=256 allows a fixed 256-row alignment (default: 128).
// Opt in when your expected per-expert token counts are >= 256 so tiles
// run at full width:
set_mk_alignment_for_contiguous_layout(
    get_theoretical_mk_alignment_for_contiguous_layout(10), // = 256 on SM100
);
```

The MoE orchestration (`moe_fp8_layer`) and both SM90/SM100 heuristics read
the same knob, so dispatch, padding and tile choice stay consistent.

```rust,no_run
# fn main() -> deepgemm::DgResult<()> {
# use deepgemm::prelude::*;
let ctx = DgContext::new(0)?;                 // a Blackwell GPU
let (m, n, k) = (4096u32, 7168u32, 7168u32);  // k must be a multiple of 128
let a: Vec<u16> = /* bf16 activations */;
let b: Vec<u16> = /* bf16 weights */;
// quantize to packed e2m1 + UE8M0 per-32 SFs, run the MXF4 kernel, download
let out = deepgemm::ops_sm100::fp4_gemm_nt_native(&ctx, &a, &b, m, n, k)?;
# let _ = out; Ok(())
# }
```

The device-level API (`fp8_fp4_gemm_dev`, `m_grouped_fp8_fp4_gemm_contiguous_dev`
(psum layout), `m_grouped_fp8_fp4_gemm_masked_dev`, `fp8_fp4_bmm_dev`) accepts
pre-quantized operands (`LpOperand`: data + packed-SF + recipe). All SM90-era
entry points (`fp8_gemm_nt*`, grouped, bmm, `bf16_gemm_nt`) automatically
route to the SM100 kernels on Blackwell — the 1d2d recipe is bridged by
transforming the FP32 scales into packed UE8M0 on the fly.

Every kernel configuration in the release matrix is validated **without a
GPU** by NVRTC+ptxas compile tests (`cargo test --test nvrtc_compile`), and
numerically on hardware by the `e2e` feature tests.

### Known SM90-only surface

MQA logits currently run the SM90 kernel path (Hopper); on Blackwell they
are not yet wired to a tcgen05 port (the upstream `sm100_mqa_logits` family
with paged/sparse schedulers is the reference for a future port).


## What's implemented

| Operator | Kernel | Notes |
|---|---|---|
| `fp8_gemm_nt` | SM90a TMA+WGMMA warp-specialized | DeepSeek recipe: per-`(1×128)` activation scales × per-`(128×128)` weight scales, FP32 promotion, BF16 out |
| `fp8_gemm_nn` / `tn` / `tt` | SMEM-tiled transpose → `nt` | layout adapters |
| `m_grouped_fp8_gemm_nt_contiguous` | same kernel, `m_indices` scheduler | MoE grouped GEMM, tokens concatenated, 128-row group alignment |
| `m_grouped_fp8_gemm_nt_masked` | same kernel, `masked_m` scheduler | DeepEP-style dispatch output, `expected_m` scheduling, zero CPU sync |
| `fp8_bmm_nt` | 3-D TMA batched variant | |
| `bf16_gemm_nt` | custom SM90a kernel **or** cuBLASLt | auto-selects cuBLASLt for large normal shapes |
| `m_grouped_bf16_gemm_nt_masked` | custom SM90a kernel | no scale factors, single-register accumulation |
| `fp4_gemm_nt` | e2m1→e4m3 unpack → FP8 engine | exact FP4→FP8 conversion kernel |
| `transform_sf` | SMEM-tiled transpose | MN-major TMA-aligned scale-factor layout (upstream `get_mn_major_tma_aligned_tensor`) |
| `mqa_logits` | block/warp-reduced dot kernel | DeepEP prefill attention logits, per-DPU valid counts on device |
| `moe_fp8_layer` | full orchestrated pipeline | histogram → offsets → gather → grouped GEMM → SiLU·mul → grouped GEMM → weighted combine, zero device→host sync |

Kernel details ported 1:1 from upstream `sm90_fp8_gemm_1d2d.cuh` (cutlass/cute
primitives rewritten as raw PTX): 384/256-thread warp specialization with
`setmaxnreg`, TMA multicast via 2-CTA clusters with L2-swizzled block scheduling,
`mbarrier` full/empty pipeline, `wgmma.mma_async` with B128-swizzled SMEM
descriptors, per-128-block scale promotion in registers, `stmatrix` + swizzled
TMA-store epilogue, and the upstream analytic config-selection cost model.

## Requirements

* Rust 1.82+
* A CUDA driver (12.x) and `libnvrtc.so` / `libnvblasLt.so` on the library path
  (any standard CUDA install; `pip install nvidia-cuda-nvrtc-cu12` also works —
  point `LD_LIBRARY_PATH` at it)
* FP8 kernels: Hopper (H100/H200, SM90a). BF16/MQA/layout kernels: SM80+.

## Quick start

```rust
use deepgemm::prelude::*;

let ctx = DgContext::new(0)?;

let (m, n, k) = (4096u32, 7168u32, 7168u32);
let a: Vec<u8> = quantize_e4m3(&random_m_by_k(m, k));       // (m, k) e4m3
let sfa = vec![0.01f32; (m * k.div_ceil(128)) as usize];    // (m, k/128)
let b: Vec<u8> = quantize_e4m3(&random_n_by_k(n, k));       // (n, k) e4m3
let sfb = vec![0.02f32; (n.div_ceil(128) * k.div_ceil(128)) as usize]; // (n/128, k/128)

let out_dev = fp8_gemm_nt(&ctx, &a, &sfa, m, &b, &sfb, n, k)?;
let out: Vec<u16> = download(&ctx, &out_dev)?;              // (m, n) bf16 bits
```

Grouped (contiguous) MoE GEMM:

```rust
// tokens of all experts concatenated; each group start 128-aligned
// b: (num_groups * n, k); sfb: (num_groups * ceil(n/128), ceil(k/128))
m_grouped_fp8_gemm_nt_contiguous_dev(
    &ctx, &a_tensor, &sfa_t, &m_indices, &b_tensor, &sfb,
    num_groups, &mut out, n,
)?;
```

## Benchmark

```sh
cargo run --release -p deepgemm-bench -- info
cargo run --release -p deepgemm-bench -- bench --op fp8_nt --m 4096 --n 7168 --k 7168
cargo run --release -p deepgemm-bench -- bench --op grouped --groups 256 --n 7168 --k 7168
cargo run --release -p deepgemm-bench -- bench --op bf16_nt --m 8192 --n 8192 --k 8192
```

Blackwell (B200/GB200) ops — every bench prints its tile decision first
(`DG_PRINT_CONFIGS` is on by default for `bench`):

```sh
# MXFP4 GEMM: row 1 = GEMM kernel only, row 2 = incl. quantize + D2H
cargo run -p deepgemm-bench --release -- bench --op fp4_nt_native --m 8192 --n 8192 --k 7168
# MoE grouped MXFP4 (contiguous / masked), 256 experts x 128 tokens
cargo run -p deepgemm-bench --release -- bench --op fp4_grouped_contig  --groups 256 --m 128 --n 7168 --k 7168
cargo run -p deepgemm-bench --release -- bench --op fp4_grouped_masked  --groups 128 --m 128 --n 2048 --k 7168
# SM100 FP8: DeepSeek gran-128 recipe and MXFP8 gran-32
cargo run -p deepgemm-bench --release -- bench --op fp8_nt_sm100   --m 4096 --n 7168 --k 7168
cargo run -p deepgemm-bench --release -- bench --op mxfp8_nt_sm100 --m 4096 --n 7168 --k 7168
cargo run -p deepgemm-bench --release -- bench --op bf16_nt_sm100  --m 8192 --n 8192 --k 8192
```

Example output (shapes will differ on your machine):

```text
[deepgemm] fp8_fp4_1d1d m=8192 n=8192 k=7168 groups=1 a_bits=4 b_bits=4 ...: block=128x256x256 stages=5 store_stages=2 cluster=1x2 swap_ab=0 swizzle=(128,128,128) smem=229232
fp4_nt_native gemm 8192x8192x7168 (kernel only)     2.117 ms   454.62 TFLOPS
fp4_nt_native (incl. quantize + D2H)                3.918 ms   245.61 TFLOPS
```

## Correctness

```sh
# CPU-side reference tests + NVRTC compile tests of every kernel (no GPU needed
# if libnvrtc is present):
LD_LIBRARY_PATH=... cargo test --workspace

# Full end-to-end GPU tests (skip automatically when the arch doesn't match):
cargo test -p deepgemm --features e2e --test e2e -- --nocapture

# CLI verify against the CPU reference:
cargo run --release -p deepgemm-bench -- verify --op fp8_nt --m 256 --n 512 --k 768
```

The `nvrtc_compile` test compiles **every** generated kernel variant (all block
sizes × gemm types × multicast modes) for `sm_90a`/`sm_100a` and runs
ptxas-level validation — it is the fastest way to catch kernel regressions in
CI without a GPU.

### B200 runbook

```sh
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH   # CUDA 12.8+

# 1) everything (e2e + TFLOPS sweep):
./scripts/b200_run.sh
# 2) or step by step:
LD_LIBRARY_PATH=... cargo test -p deepgemm --features e2e --test e2e -- --nocapture
LD_LIBRARY_PATH=... cargo run -p deepgemm-bench --release -- bench --op fp4_nt_native --m 8192 --n 8192 --k 7168
```

The e2e suite covers, on Blackwell: MXFP4 dense (`tcgen05.mma.kind::mxf4`),
MXFP4 m-grouped contiguous (psum layout) and masked, FP8 gran-128 dense, and
BF16; on Hopper it covers FP8 nt/contiguous/masked, BF16 and MQA logits.
Grouped-GEMM scale factors are verified against a CPU mirror of the exact
grouped-packed UE8M0 layout.

The `bench` output intentionally pairs every TFLOPS number with its tile
decision — if a shape underperforms, open an issue with the `[deepgemm]`
config line + the TFLOPS row and the heuristics can be tuned for that class
of shapes.

## Design notes & deliberate differences vs upstream

* **One kernel family** (`deepgemm_gemm_kernel`) is generated per configuration
  with `#define` specialization — same idea as upstream's template
  instantiation, but the "template parameters" are injected by Rust at
  source-build time, and the WGMMA inline-PTX wrappers (`m64n{N}k32.f32.e4m3.e4m3`)
  are generated programmatically for any N.
* **Runtime NVRTC + disk cache** (`~/.cache/deepgemm-rust/<sha>.ptx`): first call
  per config pays ~1s JIT; every later run loads from cache.
* **TMA descriptors are passed by value** as 64-byte-aligned kernel params via
  `cuLaunchKernelEx` (with cluster-dimension attribute for multicast configs).
* **`nn/tt/tn` layouts** go through a fast SMEM-tiled FP8 transpose instead of
  native kernels (documented trade-off: an extra bandwidth-bound pass, tiny
  relative to the GEMM at scale).
* **FP4** unpacks to FP8 first (Hopper has no FP4 tensor cores; Blackwell
  native `tcgen05` FP4 is future work).
* **MegaMoE** is an orchestrated pipeline (8 kernels, zero CPU sync) rather
  than upstream's single mega-kernel — testable, composable, and the grouped
  GEMMs it drives are the same warp-specialized kernels.
* Upstream's `k_grouped_*` (K-split with in-kernel tensormap patching),
  `fp8_gemm_nt_partial`, UE8M0/MX (SM100) scaling and PDL are **not** ported
  yet; the architecture supports adding them to the same kernel family.

## Repo layout

```
deepgemm/               the library
  src/types.rs          dtypes, layouts, operand views
  src/device.rs         context, arch caps
  src/jit.rs            NVRTC engine + disk cache
  src/tma.rs            cuTensorMapEncodeTiled builders
  src/heuristics.rs     upstream SM90 config-selection port
  src/launch.rs         cuLaunchKernelEx (clusters, 227KB smem, TMA params)
  src/cuda/             generated CUDA sources (raw PTX primitives)
    common.rs           mbarrier/TMA/WGMMA/STSM PTX helpers
    fp8_gemm_1d2d.rs    the flagship kernel + generator
    layout.rs           transform_sf, transposes, fp4 unpack, MoE dispatch
    mqa_logits.rs       MQA logits kernel
  src/ops.rs            public ops + launchers
  src/moe.rs            orchestrated MoE layer
  src/cublaslt.rs       cuBLASLt BF16 backend
  src/reference.rs      CPU reference (e4m3/bf16 decode, scaled GEMM)
  tests/nvrtc_compile.rs  kernel compile tests (no GPU needed)
  tests/e2e.rs          GPU correctness tests (feature "e2e")
deepgemm-bench/         benchmark & verification CLI
```

## License

MIT — same as upstream DeepGEMM.
