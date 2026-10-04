//! Thin cuBLASLt backend for BF16 GEMM (used for large Normal shapes where
//! cuBLASLt's heavily-tuned kernels usually edge out even a warp-specialized
//! custom kernel). Loads `libcublasLt` dynamically via cudarc's sys bindings;
//! silently unavailable on systems without it (callers fall back to the
//! custom kernel).

use std::ffi::c_void;

use cudarc::driver::safe::DevicePtr;

use cudarc::cublaslt::sys;

use crate::device::DgContext;
use crate::types::DgResult;

use cudarc::cublaslt::sys::{cublasComputeType_t, cublasLtMatmulDescAttributes_t, cudaDataType_t};

/// cublasOperation_t (lives in the cublas sys module; redefined to avoid a
/// dependency on the cublas feature).
#[repr(u32)]
#[derive(Clone, Copy)]
#[allow(non_camel_case_types)]
pub enum CublasOperation {
    OpN = 0,
    OpT = 1,
}
#[allow(non_camel_case_types)]
type cublasOperation_t = CublasOperation;

const CUDA_R_16BF: cudaDataType_t = cudaDataType_t::CUDA_R_16BF;
const CUDA_R_32F: cudaDataType_t = cudaDataType_t::CUDA_R_32F;
const CUBLAS_COMPUTE_32F: cublasComputeType_t = cublasComputeType_t::CUBLAS_COMPUTE_32F;
const TRANSA: cublasLtMatmulDescAttributes_t =
    cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA;
const TRANSB: cublasLtMatmulDescAttributes_t =
    cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB;
// cublasLtMatrixLayoutCreate / cublasLtMatmul etc. are behind sys bindings.

unsafe fn check(res: sys::cublasStatus_t) -> DgResult<()> {
    if res == sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(crate::types::DgError::Driver(format!(
            "cublasLt error {res:?}"
        )))
    }
}

/// BF16 NT GEMM: computes `out[m, n] = a[m, k] @ b[n, k]^T`.
///
/// * `a` device ptr, `(m, k)` row-major, leading dim `lda` (in elements).
/// * `b` device ptr, `(n, k)` row-major, leading dim `ldb`.
/// * `out` device ptr, `(m, n)` row-major, leading dim `ldc`.
///
/// Returns Err if cuBLASLt is unavailable.
#[allow(clippy::too_many_arguments)]
pub fn bf16_gemm_nt(
    ctx: &DgContext,
    a: u64,
    b: u64,
    out: u64,
    m: u32,
    n: u32,
    k: u32,
    lda: i64,
    ldb: i64,
    ldc: i64,
) -> DgResult<()> {
    unsafe {
        let mut handle: sys::cublasLtHandle_t = std::ptr::null_mut();
        check(sys::cublasLtCreate(&mut handle))?;

        // CUBLAS is column-major; compute out^T = b * a^T:
        //   C(n x m) = B(n x k, op=T? -> no) ... map:
        // row-major (m,k)*(k,n) == col-major (k,m)^T*(n,k)^T ... simplest:
        // col-major C' (n x m) = B' (n x k) * A' (k x m) where
        //   B' = b viewed col-major (k x n) with opB=T
        //   A' = a viewed col-major (k x m) with opA=T
        let mut desc: sys::cublasLtMatmulDesc_t = std::ptr::null_mut();
        check(sys::cublasLtMatmulDescCreate(
            &mut desc,
            CUBLAS_COMPUTE_32F,
            CUDA_R_32F,
        ))?;
        // In cublas terms: gemm(op_A, op_B, m', n', k', A, lda, B, ldb, C, ldc)
        // with m'=n, n'=m, k'=k: A = b (n x k row-major == col-major (k x n)),
        // op_A = T gives (n x k); B = a (m x k row-major == col-major (k x m)),
        // op_B = T gives (k x m). C = out (m x n row-major == col-major (n x m)).
        let op_t = CublasOperation::OpT;
        check(sys::cublasLtMatmulDescSetAttribute(
            desc,
            TRANSA,
            &op_t as *const _ as *const c_void,
            std::mem::size_of::<cublasOperation_t>(),
        ))?;
        check(sys::cublasLtMatmulDescSetAttribute(
            desc,
            TRANSB,
            &op_t as *const _ as *const c_void,
            std::mem::size_of::<cublasOperation_t>(),
        ))?;

        // Layouts (col-major): A = (k x n) with lda = ldb; B = (k x m) with
        // ldb = lda; C = (n x m) with ldc = ldc.
        let mut lay_a: sys::cublasLtMatrixLayout_t = std::ptr::null_mut();
        let mut lay_b: sys::cublasLtMatrixLayout_t = std::ptr::null_mut();
        let mut lay_c: sys::cublasLtMatrixLayout_t = std::ptr::null_mut();
        check(sys::cublasLtMatrixLayoutCreate(
            &mut lay_a,
            CUDA_R_16BF,
            k as u64,
            n as u64,
            ldb,
        ))?;
        check(sys::cublasLtMatrixLayoutCreate(
            &mut lay_b,
            CUDA_R_16BF,
            k as u64,
            m as u64,
            lda,
        ))?;
        check(sys::cublasLtMatrixLayoutCreate(
            &mut lay_c,
            CUDA_R_16BF,
            n as u64,
            m as u64,
            ldc,
        ))?;

        let mut pref: sys::cublasLtMatmulPreference_t = std::ptr::null_mut();
        check(sys::cublasLtMatmulPreferenceCreate(&mut pref))?;
        let workspace_size: usize = 32 * 1024 * 1024;
        check(sys::cublasLtMatmulPreferenceSetAttribute(
            pref,
            sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &workspace_size as *const _ as *const c_void,
            std::mem::size_of::<usize>(),
        ))?;

        let workspace = ctx
            .stream
            .alloc::<u8>(workspace_size)
            .map_err(|e| crate::types::DgError::Driver(format!("{e:?}")))?;

        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        let stream: sys::cudaStream_t = ctx.stream.cu_stream() as *mut sys::CUstream_st;

        let res = sys::cublasLtMatmul(
            handle,
            desc,
            &alpha as *const f32 as *const c_void,
            b as *const c_void, // A (col-major view of b)
            lay_a,
            a as *const c_void, // B (col-major view of a)
            lay_b,
            &beta as *const f32 as *const c_void,
            out as *const c_void, // C == D
            lay_c,
            out as *mut c_void, // D
            lay_c,
            std::ptr::null(), // algo: pick automatically
            workspace.device_ptr(&ctx.stream).0 as *mut c_void,
            workspace_size,
            stream,
        );
        let matmul_res = check(res);

        // Cleanup regardless of matmul result.
        let _ = sys::cublasLtMatmulPreferenceDestroy(pref);
        let _ = sys::cublasLtMatrixLayoutDestroy(lay_a);
        let _ = sys::cublasLtMatrixLayoutDestroy(lay_b);
        let _ = sys::cublasLtMatrixLayoutDestroy(lay_c);
        let _ = sys::cublasLtMatmulDescDestroy(desc);
        let _ = sys::cublasLtDestroy(handle);
        drop(workspace);

        matmul_res
    }
}
