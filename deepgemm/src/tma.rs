//! Host-side TMA descriptor construction (`cuTensorMapEncodeTiled`),
//! a faithful port of upstream DeepGEMM's `make_tma_2d_desc` / `make_tma_sf_desc`
//! / `make_tma_3d_desc` helpers.

use cudarc::driver::sys::{
    self, cuTensorMapEncodeTiled, CUtensorMap, CUtensorMapDataType, CUtensorMapSwizzle,
};

use crate::types::{DgError, DgResult};

pub const TMA_ELEM_FP8: CUtensorMapDataType = CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8;
pub const TMA_ELEM_BF16: CUtensorMapDataType =
    CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
pub const TMA_ELEM_F32: CUtensorMapDataType = CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT32;

fn check(res: cudarc::driver::sys::CUresult) -> DgResult<()> {
    if res == cudarc::driver::sys::CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(DgError::Driver(format!(
            "cuTensorMapEncodeTiled failed: {res:?}"
        )))
    }
}

/// Upstream `mode_into_tensor_map_swizzle`.
fn mode_to_swizzle(mode: u32) -> DgResult<CUtensorMapSwizzle> {
    Ok(match mode {
        0 | 16 => CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        32 => CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_32B,
        64 => CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_64B,
        128 => CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
        m => return Err(DgError::Shape(format!("unsupported swizzle mode {m}"))),
    })
}

/// Upstream `get_swizzle_mode`: largest swizzle atom (<= 128B) that divides
/// `inner_dim * elem_size` bytes.
pub fn get_swizzle_mode(inner_dim: u32, elem_size: u32) -> u32 {
    let bytes = inner_dim * elem_size;
    if bytes >= 128 && bytes % 128 == 0 {
        128
    } else if bytes >= 64 && bytes % 64 == 0 {
        64
    } else if bytes >= 32 && bytes % 32 == 0 {
        32
    } else {
        0
    }
}

/// Encode a 2-D tiled TMA descriptor.
///
/// * `gmem_inner`/`gmem_outer`: global tensor dims (innermost first).
/// * `gmem_outer_stride`: row stride in **elements** (must make 16B-aligned byte stride).
/// * `box_inner`/`box_outer`: shared-memory box dims.
/// * `swizzle_mode`: 0/32/64/128 bytes; if nonzero, `box_inner` is clamped to
///   `swizzle_mode / elem_size` (one atom per TMA op) — the kernel splits loops.
#[allow(clippy::too_many_arguments)]
pub fn make_tma_2d(
    addr: *const std::ffi::c_void,
    dtype: CUtensorMapDataType,
    elem_size: u32,
    gmem_inner: u32,
    gmem_outer: u32,
    gmem_outer_stride: u64,
    box_inner: u32,
    box_outer: u32,
    swizzle_mode: u32,
) -> DgResult<CUtensorMap> {
    let mut box_inner = box_inner;
    if swizzle_mode != 0 {
        box_inner = swizzle_mode / elem_size;
    }
    let _ = box_inner;

    let mut tm: CUtensorMap = unsafe { std::mem::zeroed() };
    let gdim = [gmem_inner as u64, gmem_outer as u64];
    let gstride = [gmem_outer_stride * elem_size as u64];
    let bdim = [box_inner, box_outer];
    let estride = [1u32, 1u32];

    check(unsafe {
        cuTensorMapEncodeTiled(
            &mut tm,
            dtype,
            2,
            addr as *mut _,
            gdim.as_ptr(),
            gstride.as_ptr(),
            bdim.as_ptr(),
            estride.as_ptr(),
            sys::CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
            mode_to_swizzle(swizzle_mode)?,
            sys::CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            sys::CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        )
    })?;
    Ok(tm)
}

/// Encode a 3-D tiled TMA descriptor (batched/grouped tensors).
#[allow(clippy::too_many_arguments)]
pub fn make_tma_3d(
    addr: *const std::ffi::c_void,
    dtype: CUtensorMapDataType,
    elem_size: u32,
    gmem_dims: [u32; 3],    // innermost first
    gmem_strides: [u64; 2], // in elements; [stride(dim1), stride(dim2)]
    box_dims: [u32; 3],     // smem box, innermost first
    swizzle_mode: u32,
) -> DgResult<CUtensorMap> {
    let mut box_dims = box_dims;
    if swizzle_mode != 0 {
        box_dims[0] = swizzle_mode / elem_size;
    }

    let mut tm: CUtensorMap = unsafe { std::mem::zeroed() };
    let gdim = [
        gmem_dims[0] as u64,
        gmem_dims[1] as u64,
        gmem_dims[2] as u64,
    ];
    let gstride = [
        gmem_strides[0] * elem_size as u64,
        gmem_strides[1] * elem_size as u64,
    ];
    let estride = [1u32, 1u32, 1u32];

    check(unsafe {
        cuTensorMapEncodeTiled(
            &mut tm,
            dtype,
            3,
            addr as *mut _,
            gdim.as_ptr(),
            gstride.as_ptr(),
            box_dims.as_ptr(),
            estride.as_ptr(),
            sys::CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
            mode_to_swizzle(swizzle_mode)?,
            sys::CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            sys::CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        )
    })?;
    Ok(tm)
}

/// Upstream `make_tma_sf_desc`: scale factors are stored in the transposed
/// ("MN-major TMA-aligned") layout produced by [`crate::ops::transform_sf`]:
/// a 2-D tensor `(k_blocks * num_groups, tma_aligned_mn)` with mn contiguous.
///
/// TMA box = `(block_mn, 1)` at `(m_idx, k_block_idx)`.
pub fn make_tma_sf(
    addr: *const std::ffi::c_void,
    mn: u32,
    k_blocks: u32,
    num_groups: u32,
    block_mn: u32,
) -> DgResult<CUtensorMap> {
    let aligned_mn = crate::types::tma_aligned_size(mn, 4);
    make_tma_2d(
        addr,
        TMA_ELEM_F32,
        4,
        aligned_mn,            // gmem inner dim (mn)
        k_blocks * num_groups, // gmem outer dim (k blocks x groups)
        aligned_mn as u64,     // outer stride in elements
        block_mn,              // smem box inner (mn)
        1,                     // smem box outer (one k block)
        0,                     // no swizzle for SF
    )
}
