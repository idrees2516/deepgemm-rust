//! # deepgemm
//!
//! DeepGEMM in pure Rust: fine-grained-scaled FP8 GEMM and MoE grouped GEMM
//! for NVIDIA GPUs, ported from [DeepGEMM-Ascend](https://github.com/deepseek-ai/DeepGEMM-Ascend)
//! / [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM).
//!
//! * **SM90a (Hopper)**: warp-specialized TMA + WGMMA kernels, JIT-compiled
//!   with NVRTC at runtime — no torch, no nvcc, no build step.
//! * **Fine-grained scaling**: per-`(1×128)` activation scales ×
//!   per-`(128×128)` weight scales (the DeepSeek-V3 recipe), FP32 promotion.
//! * **MoE grouped GEMM**: contiguous (`m_indices`) and masked layouts.
//! * Everything else in the family: BF16 GEMM (custom kernel + cuBLASLt),
//!   layout adapters for nn/tt/tn, FP4 unpacking, `transform_sf`, MQA
//!   logits, and an orchestrated MoE layer.
//!
//! ## Quick start
//!
//! ```no_run
//! # fn main() -> deepgemm::DgResult<()> {
//! use deepgemm::prelude::*;
//!
//! let ctx = DgContext::new(0)?;
//!
//! let (m, n, k) = (4096u32, 7168u32, 7168u32);
//! let a: Vec<u8> = quantize_e4m3(&(0..m * k).map(|i| (i % 251) as f32 / 251.0 - 0.5).collect::<Vec<_>>());
//! let b: Vec<u8> = quantize_e4m3(&(0..n * k).map(|i| (i % 241) as f32 / 241.0 - 0.5).collect::<Vec<_>>());
//! let sfa = vec![1.0f32; (m * k.div_ceil(128)) as usize];
//! let sfb = vec![1.0f32; (n.div_ceil(128) * k.div_ceil(128)) as usize];
//!
//! let out = fp8_gemm_nt(&ctx, &a, &sfa, m, &b, &sfb, n, k)?;
//! # let _ = out;
//! # Ok(())
//! # }
//! ```
//!
//! Kernels are compiled on first use and cached on disk
//! (`~/.cache/deepgemm-rust`), so subsequent runs start instantly.

pub mod cuda;
pub mod device;
pub mod heuristics;
pub mod heuristics_sm100;
pub mod jit;
pub mod launch;
pub mod ops;
pub mod ops_sm100;
pub mod reference;
pub mod tma;
pub mod types;

pub mod cublaslt;
pub mod moe;

pub mod prelude {
    pub use crate::device::{Arch, DgContext};
    pub use crate::ops::*;
    pub use crate::reference::quantize_e4m3;
    pub use crate::types::{
        Bf16Operand, DgError, DgResult, Fp8Operand, GemmType, OutDType, OutTensor,
    };
}

pub use types::{DgError, DgResult};
