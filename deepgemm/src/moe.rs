//! Orchestrated MoE layer (the "MegaMoE" surface of DeepGEMM-Ascend) as a
//! zero-CPU-synchronization pipeline:
//!
//! 1. expert histogram (device atomics)
//! 2. aligned per-expert offsets (device)
//! 3. per-token destination slots (device atomics)
//! 4. gather token rows + scale rows into the contiguous grouped layout
//! 5. grouped GEMM #1: `(gate|up)` projection (contiguous layout)
//! 6. `silu(gate) * up` activation
//! 7. grouped GEMM #2: down projection
//! 8. weighted scatter-add back to token order
//!
//! The only host-side knowledge needed is an upper bound on the padded token
//! count (`num_tokens + num_experts * 127`), so no device-to-host
//! synchronization is ever required for dynamic routing.

use std::sync::Arc;

use cudarc::driver::safe::{CudaModule, CudaSlice};

use crate::device::DgContext;
use crate::launch::{ArgBuilder, DevPtr, LaunchGrid};
use crate::ops::{dev_ptr, Fp8Tensor};
use crate::types::{DgError, DgResult, MK_ALIGNMENT_FOR_CONTIGUOUS_LAYOUT as ALIGN};

fn layout_module(ctx: &DgContext) -> DgResult<Arc<CudaModule>> {
    ctx.jit.module(
        &ctx.device,
        &crate::cuda::layout::build_layout_kernel_source(),
    )
}

fn launch1(
    ctx: &DgContext,
    module: &Arc<CudaModule>,
    name: &str,
    args: &mut ArgBuilder,
    grid_x: u32,
    block: u32,
) -> DgResult<()> {
    let func = module
        .load_function(name)
        .map_err(|e| DgError::Driver(format!("{e:?}")))?;
    unsafe { args.launch(&func, &ctx.stream, LaunchGrid::new(grid_x, block, 0)) }
}

/// Everything the MoE pipeline needs to know about the weights.
pub struct MoeWeights<'a> {
    /// `(num_experts * 2 * hidden, k)` fp8 — expert `g`'s gate_up rows at
    /// `[g * 2 * hidden, (g + 1) * 2 * hidden)`.
    pub gate_up: &'a CudaSlice<u8>,
    pub gate_up_sf: &'a CudaSlice<f32>,
    /// `(num_experts * hidden, k)` fp8.
    pub down: &'a CudaSlice<u8>,
    pub down_sf: &'a CudaSlice<f32>,
    pub hidden: u32,
    pub k: u32,
    pub num_experts: u32,
}

/// Gather `row_bytes`-sized rows of `src` into `dst` at `dst_pos[t]`
/// (byte-addressable via the u8 gather kernel).
fn gather_rows_bytes(
    ctx: &DgContext,
    module: &Arc<CudaModule>,
    src: &CudaSlice<u8>,
    dst: &CudaSlice<u8>,
    dst_pos: &CudaSlice<i32>,
    num_tokens: u32,
    row_bytes: u32,
) -> DgResult<()> {
    let total = num_tokens as u64 * row_bytes as u64;
    let mut args = ArgBuilder::new();
    args.push(&DevPtr(dev_ptr(ctx, src)));
    args.push(&DevPtr(dev_ptr(ctx, dst)));
    args.push(&DevPtr(dev_ptr(ctx, dst_pos)));
    args.push(&num_tokens);
    args.push(&row_bytes);
    launch1(
        ctx,
        module,
        "deepgemm_gather_rows_fp8",
        &mut args,
        total.div_ceil(256) as u32,
        256,
    )
}

/// Run a complete FP8 MoE layer and return `(num_tokens, hidden)` bf16.
///
/// * `tokens`: `(num_tokens, k)` fp8 e4m3, K-major.
/// * `tokens_sf`: `(num_tokens, ceil(k / 128))` f32.
/// * `expert_ids`: `(num_tokens)` i32 in `[0, num_experts)`.
/// * `combine_weights`: `(num_tokens)` f32.
#[allow(clippy::too_many_arguments)]
pub fn moe_fp8_layer(
    ctx: &DgContext,
    tokens: &CudaSlice<u8>,
    tokens_sf: &CudaSlice<f32>,
    expert_ids: &CudaSlice<i32>,
    combine_weights: &CudaSlice<f32>,
    num_tokens: u32,
    weights: &MoeWeights,
) -> DgResult<CudaSlice<u16>> {
    let (k, hidden, num_experts) = (weights.k, weights.hidden, weights.num_experts);
    let k_blocks = k.div_ceil(128);
    let sf_row_bytes = k_blocks * 4;

    // Worst-case padded rows (every group 128-aligned); never read on host.
    let padded_m = num_tokens + num_experts * (ALIGN - 1);
    let lk = layout_module(ctx)?;

    // ---------------------------------------------------------- dispatch
    let counts = ctx
        .stream
        .alloc_zeros::<i32>(num_experts as usize)
        .map_err(|e| DgError::Driver(format!("{e:?}")))?;
    let offsets = ctx
        .stream
        .alloc_zeros::<i32>(num_experts as usize + 1)
        .map_err(|e| DgError::Driver(format!("{e:?}")))?;
    let slots = ctx
        .stream
        .alloc_zeros::<i32>(num_experts as usize)
        .map_err(|e| DgError::Driver(format!("{e:?}")))?;
    let dst_pos = ctx
        .stream
        .alloc_zeros::<i32>(num_tokens as usize)
        .map_err(|e| DgError::Driver(format!("{e:?}")))?;

    {
        let mut a = ArgBuilder::new();
        a.push(&DevPtr(dev_ptr(ctx, expert_ids)));
        a.push(&DevPtr(dev_ptr(ctx, &counts)));
        a.push(&num_tokens);
        launch1(
            ctx,
            &lk,
            "deepgemm_expert_histogram",
            &mut a,
            num_tokens.div_ceil(256),
            256,
        )?;
    }
    {
        let mut a = ArgBuilder::new();
        a.push(&DevPtr(dev_ptr(ctx, &counts)));
        a.push(&DevPtr(dev_ptr(ctx, &offsets)));
        a.push(&num_experts);
        a.push(&ALIGN);
        launch1(ctx, &lk, "deepgemm_permute_offsets", &mut a, 1, 32)?;
    }
    {
        let mut a = ArgBuilder::new();
        a.push(&DevPtr(dev_ptr(ctx, expert_ids)));
        a.push(&DevPtr(dev_ptr(ctx, &offsets)));
        a.push(&DevPtr(dev_ptr(ctx, &slots)));
        a.push(&DevPtr(dev_ptr(ctx, &dst_pos)));
        a.push(&num_tokens);
        launch1(
            ctx,
            &lk,
            "deepgemm_assign_dst_pos",
            &mut a,
            num_tokens.div_ceil(256),
            256,
        )?;
    }

    // Gather token rows + SF rows into the grouped contiguous layout.
    let a_grouped = unsafe {
        ctx.stream
            .alloc::<u8>(padded_m as usize * k as usize)
            .map_err(|e| DgError::Driver(format!("{e:?}")))?
    };
    let sf_grouped = unsafe {
        ctx.stream
            .alloc::<f32>(padded_m as usize * k_blocks as usize)
            .map_err(|e| DgError::Driver(format!("{e:?}")))?
    };
    gather_rows_bytes(ctx, &lk, tokens, &a_grouped, &dst_pos, num_tokens, k)?;

    // SF gather: source is `tokens_sf` (rows of k_blocks f32), destination is
    // `sf_grouped`; the kernel works on raw bytes with `sf_row_bytes` rows.
    {
        let total = num_tokens as u64 * sf_row_bytes as u64;
        let mut a = ArgBuilder::new();
        a.push(&DevPtr(dev_ptr(ctx, tokens_sf)));
        a.push(&DevPtr(dev_ptr(ctx, &sf_grouped)));
        a.push(&DevPtr(dev_ptr(ctx, &dst_pos)));
        a.push(&num_tokens);
        a.push(&sf_row_bytes);
        launch1(
            ctx,
            &lk,
            "deepgemm_gather_rows_fp8",
            &mut a,
            total.div_ceil(256) as u32,
            256,
        )?;
    }

    // m_indices: -1 for padding rows, expert id on valid rows.
    let m_indices = ctx
        .stream
        .alloc_zeros::<i32>(padded_m as usize)
        .map_err(|e| DgError::Driver(format!("{e:?}")))?;
    {
        let mut a = ArgBuilder::new();
        a.push(&DevPtr(dev_ptr(ctx, &m_indices)));
        let n = padded_m as u64;
        a.push(&n);
        let v: i32 = -1;
        a.push(&v);
        launch1(
            ctx,
            &lk,
            "deepgemm_fill_i32",
            &mut a,
            n.div_ceil(256) as u32,
            256,
        )?;
    }
    {
        let mut a = ArgBuilder::new();
        a.push(&DevPtr(dev_ptr(ctx, &dst_pos)));
        a.push(&DevPtr(dev_ptr(ctx, expert_ids)));
        a.push(&DevPtr(dev_ptr(ctx, &m_indices)));
        a.push(&num_tokens);
        a.push(&padded_m);
        launch1(
            ctx,
            &lk,
            "deepgemm_build_m_indices_from_pos",
            &mut a,
            num_tokens.div_ceil(256),
            256,
        )?;
    }

    // ------------------------------------------------------- grouped GEMM 1
    let mut gate_up_out = unsafe {
        ctx.stream
            .alloc::<u16>(padded_m as usize * (2 * hidden) as usize)
            .map_err(|e| DgError::Driver(format!("{e:?}")))?
    };
    let a_tensor = Fp8Tensor::new(&a_grouped, &sf_grouped, padded_m, k);
    let sfa_t = crate::ops::transform_sf_dev(ctx, &sf_grouped, padded_m, k_blocks, 1)?;
    let b_gu = Fp8Tensor::new(
        weights.gate_up,
        weights.gate_up_sf,
        num_experts * 2 * hidden,
        k,
    );
    crate::ops::m_grouped_fp8_gemm_nt_contiguous_dev(
        ctx,
        &a_tensor,
        &sfa_t,
        &m_indices,
        &b_gu,
        weights.gate_up_sf,
        num_experts,
        &mut gate_up_out,
        2 * hidden,
    )?;

    // ------------------------------------------------------- silu * up
    let act_out = unsafe {
        ctx.stream
            .alloc::<u16>(padded_m as usize * hidden as usize)
            .map_err(|e| DgError::Driver(format!("{e:?}")))?
    };
    {
        let total = padded_m as u64 * hidden as u64;
        let mut a = ArgBuilder::new();
        a.push(&DevPtr(dev_ptr(ctx, &gate_up_out)));
        a.push(&DevPtr(dev_ptr(ctx, &act_out)));
        a.push(&hidden);
        a.push(&total);
        launch1(
            ctx,
            &lk,
            "deepgemm_silu_mul_bf16",
            &mut a,
            total.div_ceil(256) as u32,
            256,
        )?;
    }

    // ------------------------------------------------------- grouped GEMM 2
    let mut down_out = unsafe {
        ctx.stream
            .alloc::<u16>(padded_m as usize * hidden as usize)
            .map_err(|e| DgError::Driver(format!("{e:?}")))?
    };
    let b_down = Fp8Tensor::new(weights.down, weights.down_sf, num_experts * hidden, k);
    crate::ops::m_grouped_fp8_gemm_nt_contiguous_dev(
        ctx,
        &a_tensor,
        &sfa_t,
        &m_indices,
        &b_down,
        weights.down_sf,
        num_experts,
        &mut down_out,
        hidden,
    )?;

    // ------------------------------------------------------- combine
    let acc = ctx
        .stream
        .alloc_zeros::<f32>(num_tokens as usize * hidden as usize)
        .map_err(|e| DgError::Driver(format!("{e:?}")))?;
    {
        // Rows of `down_out` are valid iff their m_indices entry >= 0; the
        // scatter kernel takes a dst_pos per src row. Build the inverse map:
        // valid rows' positions are exactly dst_pos values; iterate tokens.
        let total = num_tokens as u64 * hidden as u64;
        let mut a = ArgBuilder::new();
        a.push(&DevPtr(dev_ptr(ctx, &down_out)));
        a.push(&DevPtr(dev_ptr(ctx, &acc)));
        a.push(&DevPtr(dev_ptr(ctx, &dst_pos))); // per-token src row
        a.push(&DevPtr(dev_ptr(ctx, combine_weights)));
        a.push(&num_tokens); // src rows enumerated by token
        a.push(&hidden);
        launch1(
            ctx,
            &lk,
            "deepgemm_scatter_rows_f32_acc",
            &mut a,
            total.div_ceil(256) as u32,
            256,
        )?;
    }
    let out = unsafe {
        ctx.stream
            .alloc::<u16>(num_tokens as usize * hidden as usize)
            .map_err(|e| DgError::Driver(format!("{e:?}")))?
    };
    {
        let total = num_tokens as u64 * hidden as u64;
        let mut a = ArgBuilder::new();
        a.push(&DevPtr(dev_ptr(ctx, &acc)));
        a.push(&DevPtr(dev_ptr(ctx, &out)));
        a.push(&total);
        launch1(
            ctx,
            &lk,
            "deepgemm_f32_to_bf16",
            &mut a,
            total.div_ceil(256) as u32,
            256,
        )?;
    }

    Ok(out)
}
