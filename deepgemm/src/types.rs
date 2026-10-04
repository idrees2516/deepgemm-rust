//! Core data types: GEMM kinds, layouts, dtypes and tensor views.
//!
//! Mirrors the upstream DeepGEMM / DeepGEMM-Ascend API surface, but without
//! any torch dependency: tensors are lightweight borrowed views over
//! device memory allocated by [`cudarc`].

use thiserror::Error;

#[derive(Debug, Error)]
pub enum DgError {
    #[error("CUDA driver error: {0}")]
    Driver(String),
    #[error("NVRTC compile error:\n{0}")]
    Nvrtc(String),
    #[error("shape/layout violation: {0}")]
    Shape(String),
    #[error("unsupported operation on this device: {0}")]
    Unsupported(String),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

pub type DgResult<T> = Result<T, DgError>;

/// Which GEMM problem the kernel is solving (mirrors `deep_gemm::GemmType`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GemmType {
    /// Plain `D[m,n] = A[m,k] @ B[n,k]^T`.
    Normal,
    /// MoE grouped GEMM, tokens of all experts concatenated along M.
    /// `m_indices[i]` = expert id of row `i`, or `-1` for padding rows.
    MGroupedContiguous,
    /// MoE grouped GEMM with per-expert fixed M padding.
    /// A/B/D are 3-D `(num_groups, m, ...)`, `masked_m[g]` = valid rows.
    MGroupedMasked,
    /// Batched (bmm-style) grouped GEMM.
    Batched,
}

/// Memory major-ness of an operand's *inner* (contiguous) dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Major {
    /// K (reduction) dimension is contiguous — the DeepGEMM-native layout.
    K,
    /// M/N dimension is contiguous.
    Mn,
}

/// Output element type of the GEMM epilogue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OutDType {
    BFloat16,
    Float16,
    Float32,
}

/// A borrowed FP8 (e4m3) operand: data + per-128-channel scale factors.
///
/// * `data`: `(rows, k)` e4m3 bytes, K-major (k contiguous), row stride `ld` bytes.
/// * `sf`: `(rows, ceil(k / 128))` f32 scale factors, K contiguous
///   (DeepGEMM "1d" recipe for activations, `(1, 128)` granularity).
/// * For weights with 2-D scaling (`(128, 128)` blocks, the DeepSeek recipe):
///   `sf` has shape `(ceil(rows / 128), ceil(k / 128))`, K contiguous.
#[derive(Debug, Clone, Copy)]
pub struct Fp8Operand<'a> {
    pub data: &'a [u8],
    pub sf: &'a [f32],
    pub rows: u32,
    pub k: u32,
    /// Row stride of `data` in **elements** (>= k). Must be multiple of 16.
    pub ld: u32,
}

impl<'a> Fp8Operand<'a> {
    pub fn new(data: &'a [u8], sf: &'a [f32], rows: u32, k: u32) -> DgResult<Self> {
        Self::with_ld(data, sf, rows, k, k)
    }

    pub fn with_ld(data: &'a [u8], sf: &'a [f32], rows: u32, k: u32, ld: u32) -> DgResult<Self> {
        if data.len() < rows as usize * ld as usize {
            return Err(DgError::Shape("fp8 data buffer too small".into()));
        }
        if ld % 16 != 0 {
            return Err(DgError::Shape(
                "fp8 leading dimension must be a multiple of 16".into(),
            ));
        }
        let sf_cols = k.div_ceil(128);
        if sf.len() < rows as usize * sf_cols as usize {
            return Err(DgError::Shape(format!(
                "sf buffer too small: {sf_len} < {rows} * {sf_cols}",
                sf_len = sf.len()
            )));
        }
        Ok(Self {
            data,
            sf,
            rows,
            k,
            ld,
        })
    }
}

/// A borrowed BF16/FP16 operand `(rows, k)`, K-major.
#[derive(Debug, Clone, Copy)]
pub struct Bf16Operand<'a> {
    pub data: &'a [u16],
    pub rows: u32,
    pub k: u32,
    pub ld: u32,
}

impl<'a> Bf16Operand<'a> {
    pub fn new(data: &'a [u16], rows: u32, k: u32) -> DgResult<Self> {
        Self::with_ld(data, rows, k, k)
    }

    pub fn with_ld(data: &'a [u16], rows: u32, k: u32, ld: u32) -> DgResult<Self> {
        if data.len() < rows as usize * ld as usize {
            return Err(DgError::Shape(format!(
                "bf16 data buffer too small: {data_len} < {rows} * {ld}",
                data_len = data.len()
            )));
        }
        Ok(Self { data, rows, k, ld })
    }
}

/// A mutable output view `(m, n)`, row-major, element type `u16` (bf16/f16)
/// or `f32`.
#[derive(Debug)]
pub struct OutTensor<'a> {
    pub data: &'a mut [u16],
    pub rows: u32,
    pub cols: u32,
    pub ld: u32,
    pub dtype: OutDType,
}

impl<'a> OutTensor<'a> {
    pub fn bf16(data: &'a mut [u16], rows: u32, cols: u32) -> DgResult<Self> {
        Self::bf16_with_ld(data, rows, cols, cols)
    }

    pub fn bf16_with_ld(data: &'a mut [u16], rows: u32, cols: u32, ld: u32) -> DgResult<Self> {
        if data.len() < rows as usize * ld as usize {
            return Err(DgError::Shape(format!(
                "output buffer too small: {len} < {rows} * {ld}",
                len = data.len()
            )));
        }
        Ok(Self {
            data,
            rows,
            cols,
            ld,
            dtype: OutDType::BFloat16,
        })
    }

    pub fn f16_with_ld(data: &'a mut [u16], rows: u32, cols: u32, ld: u32) -> DgResult<Self> {
        let mut t = Self::bf16_with_ld(data, rows, cols, ld)?;
        t.dtype = OutDType::Float16;
        Ok(t)
    }
}

/// A `(rows, cols)` f32 buffer with row stride (scale factors, partial sums).
#[derive(Debug)]
pub struct F32Tensor<'a> {
    pub data: &'a mut [f32],
    pub rows: u32,
    pub cols: u32,
    pub ld: u32,
}

/// Helper: the M/N alignment (in rows) required by contiguous grouped GEMM
/// (upstream `get_mk_alignment_for_contiguous_layout`).
pub const MK_ALIGNMENT_FOR_CONTIGUOUS_LAYOUT: u32 = 128;

/// TMA alignment for scale factor tensors: pad the MN dim so each k-column
/// of the transposed SF tensor is 16 bytes (4 floats).
pub const fn tma_aligned_size(mn: u32, elem_size: u32) -> u32 {
    mn.div_ceil(16 / elem_size) * (16 / elem_size)
}
