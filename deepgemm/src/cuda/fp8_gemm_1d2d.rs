//! The flagship kernel: SM90a FP8 GEMM with fine-grained scaling ("1d2d":
//! per-row-per-128-k activation scales × per-128×128-tile weight scales),
//! warp-specialized TMA + WGMMA, ported 1:1 from upstream DeepGEMM's
//! `sm90_fp8_gemm_1d2d.cuh` (cutlass/cute primitives replaced by raw PTX).
//!
//! Covers: Normal, MGroupedContiguous (m_indices), MGroupedMasked, Batched.

use super::common::COMMON_HEADER;

/// Generate the `wgmma_m64n{N}k32.f32.e4m3.e4m3` wrapper for a given N.
fn gen_wgmma_fp8(n: u32) -> String {
    let nregs = (n / 2) as usize;
    let regs: Vec<String> = (0..nregs).map(|i| format!("%{i}")).collect();
    let outs: Vec<String> = (0..nregs).map(|i| format!("\"+f\"(d[{i}])")).collect();
    let regs = regs.join(",");
    let outs = outs.join(", ");
    let ia = nregs; // operand index of desc_a
    let ib = nregs + 1; // desc_b
    let isd = nregs + 2; // scale_d immediate
    format!(
        r#"
template <bool SD>
DG_INLINE void dg_wgmma_fp8(float* d, uint64_t desc_a, uint64_t desc_b) {{
    asm volatile(
        "{{\n"
        "wgmma.mma_async.sync.aligned.m64n{n}k32.f32.e4m3.e4m3 "
        "{{{regs}}}, %{ia}, %{ib}, %{isd}, 1, 1;\n"
        "}}\n"
        : {outs}
        : "l"(desc_a), "l"(desc_b), "n"(SD ? 1 : 0));
}}
"#
    )
}

/// Generate the `wgmma_m64n{N}k16.f32.bf16.bf16` wrapper (K-major A and B).
fn gen_wgmma_bf16(n: u32) -> String {
    let nregs = (n / 2) as usize;
    let regs: Vec<String> = (0..nregs).map(|i| format!("%{i}")).collect();
    let outs: Vec<String> = (0..nregs).map(|i| format!("\"+f\"(d[{i}])")).collect();
    let regs = regs.join(",");
    let outs = outs.join(", ");
    let ia = nregs; // operand index of desc_a
    let ib = nregs + 1; // desc_b
    let isd = nregs + 2; // scale_d immediate
    format!(
        r#"
template <bool SD>
DG_INLINE void dg_wgmma_bf16(float* d, uint64_t desc_a, uint64_t desc_b) {{
    asm volatile(
        "{{\n"
        "wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 "
        "{{{regs}}}, %{ia}, %{ib}, %{isd}, 1, 1, 0, 0;\n"
        "}}\n"
        : {outs}
        : "l"(desc_a), "l"(desc_b), "n"(SD ? 1 : 0));
}}
"#
    )
}

/// The kernel template. `%%VAR%%` placeholders are substituted per config.
const KERNEL_TEMPLATE: &str = r#"
// ===========================================================================
// deepgemm-rust :: sm90_fp8_gemm_1d2d (generated for one block configuration)
// Port of upstream DeepGEMM sm90_fp8_gemm_1d2d.cuh
// ===========================================================================
#define BLOCK_M             %%BLOCK_M%%
#define BLOCK_N             %%BLOCK_N%%
#define BLOCK_K             %%BLOCK_K%%
#define NUM_STAGES          %%NUM_STAGES%%
#define SWIZZLE_A           %%SWIZZLE_A%%       // bytes
#define SWIZZLE_B           %%SWIZZLE_B%%
#define SWIZZLE_D           %%SWIZZLE_D%%
#define NUM_TMA_THREADS     %%NUM_TMA_THREADS%%
#define NUM_MATH_THREADS    %%NUM_MATH_THREADS%%
#define MULTICAST           %%MULTICAST%%
#define IS_MULTICAST_ON_A   %%IS_MULTICAST_ON_A%%   // 0/1
#define NUM_SMS             %%NUM_SMS%%
#define GEMM_TYPE           %%GEMM_TYPE%%   // 0=Normal 1=MGroupedContiguous 2=MGroupedMasked 3=Batched
#define IS_BF16             %%IS_BF16%%     // 0 = fp8+SF, 1 = bf16, no SF
#define NUM_THREADS         (NUM_TMA_THREADS + NUM_MATH_THREADS)
#define IS_3D_TMA           (GEMM_TYPE == 3)

// WGMMA shape
#define WGMMA_M 64
#define WGMMA_K (IS_BF16 ? 16 : 32)
#define NUM_ACCUM (WGMMA_M * BLOCK_N / 128)
    #define WAVE_BLOCK_M (BLOCK_M <= WGMMA_M ? BLOCK_M : WGMMA_M * 2)
#define NUM_WGMMA_PER_K (BLOCK_K / WGMMA_K)
#define NUM_WGMMA_STORE_THREADS (WAVE_BLOCK_M * (128 / WGMMA_M))
#define TMA_D_BLOCK_N (SWIZZLE_D == 0 ? BLOCK_N : SWIZZLE_D / 2)   // bf16 elements
#define MUST_USE_UNIFORMED_SCALE_B (IS_BF16 || (BLOCK_K % BLOCK_N == 0))

#if IS_BF16
    typedef uint16_t ab_dtype_t;
    #define AB_ELEM_SIZE 2
#else
    typedef uint8_t ab_dtype_t;
    #define AB_ELEM_SIZE 1
#endif

#if !IS_BF16
static_assert(BLOCK_K == 128, "per-128-channel FP8 scaling requires BLOCK_K == 128");
static_assert(CEIL_DIV_CONST(BLOCK_N, BLOCK_K) == 1 || CONST_GCD(BLOCK_N, BLOCK_K) == BLOCK_N - BLOCK_K,
              "too much B scales in a single block");
#endif

extern "C" __global__ void __launch_bounds__(NUM_THREADS, 1) deepgemm_gemm_kernel(
#if !IS_BF16
        float* sfb,                    // (num_groups*n_tiles, k_blocks) or transposed
#endif
        int* grouped_layout,           // m_indices (contiguous) / masked_m (masked) / null
        uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
        const __grid_constant__ TmaDescriptor tensor_map_a,
        const __grid_constant__ TmaDescriptor tensor_map_b,
        const __grid_constant__ TmaDescriptor tensor_map_d
#if !IS_BF16
        , const __grid_constant__ TmaDescriptor tensor_map_sfa
#endif
) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    // ---------------------------------------------------------------- setup
    const uint32_t warp_idx = dg_warp_idx();
    const uint32_t lane_idx = dg_lane_idx();

    // SMEM layout (all offsets in bytes):
    //   [D tile][A stages][B stages][SFA stages][SFB staging][barriers]
#if !IS_BF16
    constexpr uint32_t SMEM_SFA_PER_STAGE = BLOCK_M * 4;
    constexpr uint32_t ALIGNED_SMEM_SFA_PER_STAGE = (SMEM_SFA_PER_STAGE + 127) / 128 * 128;
#else
    constexpr uint32_t SMEM_SFA_PER_STAGE = 0;
    constexpr uint32_t ALIGNED_SMEM_SFA_PER_STAGE = 0;
#endif
    constexpr uint32_t SMEM_A_PER_STAGE = BLOCK_M * BLOCK_K * AB_ELEM_SIZE;
    constexpr uint32_t SMEM_B_PER_STAGE = BLOCK_N * BLOCK_K * AB_ELEM_SIZE;
    constexpr uint32_t SMEM_D_SIZE = (BLOCK_M * BLOCK_N * 2 + 1023) / 1024 * 1024;

#if !IS_BF16
    const uint32_t shape_k_scales = dg_ceil_div(shape_k, BLOCK_K);
    const uint32_t shape_n_sfb = dg_ceil_div(shape_n, BLOCK_K);
    #if MUST_USE_UNIFORMED_SCALE_B
    const uint32_t smem_sfb_size = (shape_k_scales * 4 + 7) / 8 * 8;
    #else
    const uint32_t smem_sfb_size = (shape_k_scales * 2 * 4 + 7) / 8 * 8;
    #endif
#else
    const uint32_t shape_k_scales = 0;
    const uint32_t shape_n_sfb = 0;
    const uint32_t smem_sfb_size = 0;
#endif
    constexpr uint32_t SMEM_SF_OFFSET = SMEM_D_SIZE + NUM_STAGES * (SMEM_A_PER_STAGE + SMEM_B_PER_STAGE);

    // WGMMA requires enough A padding slack (upstream memory bound assert)
    static_assert(WGMMA_M * BLOCK_K * AB_ELEM_SIZE <= SMEM_A_PER_STAGE + SMEM_B_PER_STAGE * NUM_STAGES,
                  "out of shared memory bound for WGMMA");
    static_assert(SMEM_D_SIZE % 1024 == 0, "D smem must be 1024-aligned");
    static_assert(BLOCK_N % 8 == 0, "invalid swizzling atom");
    static_assert(BLOCK_N % TMA_D_BLOCK_N == 0 && BLOCK_N / TMA_D_BLOCK_N <= 32, "unaligned TMA store");
    static_assert(TMA_D_BLOCK_N % 8 == 0, "invalid TMA D block N");

    extern __shared__ __align__(1024) uint8_t smem_buffer[];

    uint16_t* smem_d = reinterpret_cast<uint16_t*>(smem_buffer);
    auto smem_a = [&](uint32_t s) -> ab_dtype_t* {
        return reinterpret_cast<ab_dtype_t*>(smem_buffer + SMEM_D_SIZE + s * SMEM_A_PER_STAGE);
    };
    auto smem_b = [&](uint32_t s) -> ab_dtype_t* {
        return reinterpret_cast<ab_dtype_t*>(smem_buffer + SMEM_D_SIZE + NUM_STAGES * SMEM_A_PER_STAGE + s * SMEM_B_PER_STAGE);
    };
#if !IS_BF16
    auto smem_sfa = [&](uint32_t s) -> float* {
        return reinterpret_cast<float*>(smem_buffer + SMEM_SF_OFFSET + s * ALIGNED_SMEM_SFA_PER_STAGE);
    };
    float* smem_sfb = reinterpret_cast<float*>(smem_buffer + SMEM_SF_OFFSET + NUM_STAGES * ALIGNED_SMEM_SFA_PER_STAGE);
#endif

    // Barriers (8 bytes each)
    uint8_t* barrier_start =
#if !IS_BF16
        reinterpret_cast<uint8_t*>(smem_sfb) + smem_sfb_size;
#else
        smem_buffer + SMEM_SF_OFFSET;
#endif
    auto full_barrier = [&](uint32_t s) -> void* { return barrier_start + s * 8; };
    auto empty_barrier = [&](uint32_t s) -> void* { return barrier_start + (NUM_STAGES + s) * 8; };

    // ---------------------------------------------------- init + prefetch
    if (warp_idx == NUM_MATH_THREADS / 32 && dg_elect_one()) {
        dg_prefetch_tma_descriptor(&tensor_map_a);
        dg_prefetch_tma_descriptor(&tensor_map_b);
        dg_prefetch_tma_descriptor(&tensor_map_d);
#if !IS_BF16
        dg_prefetch_tma_descriptor(&tensor_map_sfa);
#endif
    }
    __syncwarp();

    if (warp_idx == NUM_MATH_THREADS / 32 + 1 && dg_elect_one()) {
        #pragma unroll
        for (uint32_t s = 0; s < NUM_STAGES; ++s) {
            dg_mbarrier_init(full_barrier(s), 1);
            dg_mbarrier_init(empty_barrier(s), MULTICAST * NUM_MATH_THREADS / 32);
        }
        dg_fence_barrier_init();
    }
#if MULTICAST > 1
    dg_cluster_sync_relaxed();
#else
    __syncthreads();
#endif

    // Register reallocation
    dg_grid_dependency_sync();

    // ------------------------------------------------------- scheduler
    // (inlined port of deep_gemm::sched::Scheduler for the supported types)
    const uint32_t num_m_blocks_total = dg_ceil_div(shape_m, BLOCK_M);
    const uint32_t num_n_blocks = dg_ceil_div(shape_n, BLOCK_N);

    uint32_t num_m_blocks = num_m_blocks_total;
    uint32_t num_blocks = num_m_blocks * num_n_blocks;
    uint32_t num_blocks_in_group = 1;
    bool is_peer_cta_alive = true;

    int scheduler_iter = -1;
    uint32_t current_group_idx = 0;
    uint32_t current_m_cumsum = 0;
    uint32_t num_1d_blocks_per_group = (MULTICAST > 1 ? 8u : 16u);

    // L2 swizzle (upstream get_swizzled_block_idx)
    auto get_swizzled_block_idx = [&](uint32_t block_idx, uint32_t& m_block_idx, uint32_t& n_block_idx) {
        const bool is_mc_on_a = IS_MULTICAST_ON_A;
        const uint32_t primary_num_blocks = is_mc_on_a ? num_n_blocks : num_m_blocks;
        const uint32_t secondary_num_blocks = is_mc_on_a ? num_m_blocks : num_n_blocks;
        const uint32_t num_blocks_per_group = secondary_num_blocks * num_1d_blocks_per_group;
        const uint32_t group_idx = block_idx / num_blocks_per_group;
        uint32_t first_block_idx = group_idx * num_1d_blocks_per_group;
        uint32_t in_group_idx = block_idx % num_blocks_per_group;
        num_blocks_in_group = dg_min_u32(num_1d_blocks_per_group, primary_num_blocks - first_block_idx);

        // Fix unaligned TMA multicast
        if (MULTICAST > 1 && num_blocks_in_group % 2 != 0) {
            if (in_group_idx < (num_blocks_in_group ^ 1) * secondary_num_blocks) {
                num_blocks_in_group = num_blocks_in_group ^ 1;
            } else {
                in_group_idx -= (num_blocks_in_group ^ 1) * secondary_num_blocks;
                first_block_idx += num_blocks_in_group ^ 1;
                num_blocks_in_group = 1;
            }
        }

        if (is_mc_on_a) {
            m_block_idx = in_group_idx / num_blocks_in_group;
            n_block_idx = first_block_idx + in_group_idx % num_blocks_in_group;
        } else {
            m_block_idx = first_block_idx + in_group_idx % num_blocks_in_group;
            n_block_idx = in_group_idx / num_blocks_in_group;
        }
    };

    // group offset (in elements) for the given index type
    // IndexType: 0 = MN, 1 = K, 2 = SF_K
    auto get_global_idx = [&](bool with_group_offset, int index_type,
                              uint32_t shape_dim, uint32_t block_size,
                              uint32_t block_idx, uint32_t m_block_idx) -> uint32_t {
        uint32_t offset = 0;
        if (with_group_offset) {
            if (GEMM_TYPE == 0) {           // Normal
                offset = 0;
            } else if (GEMM_TYPE == 1) {    // MGroupedContiguous
                offset = dg_max_i32(0, grouped_layout[m_block_idx * BLOCK_M]);
            } else if (GEMM_TYPE == 2) {    // MGroupedMasked
                offset = current_group_idx;
            } else if (GEMM_TYPE == 3) {    // Batched: only SF_K carries a group offset
                offset = (index_type == 2) ? current_group_idx : 0;
            }
        }
        return offset * shape_dim + block_idx * block_size;
    };

    auto get_next_block = [&](uint32_t& m_block_idx, uint32_t& n_block_idx) -> bool {
        const uint32_t next_block_idx = (++scheduler_iter) * NUM_SMS + blockIdx.x;

        if (GEMM_TYPE == 2) {               // MGroupedMasked
            while (true) {
                if (current_group_idx >= NUM_GROUPS)
                    return false;
                num_m_blocks = dg_ceil_div(static_cast<uint32_t>(grouped_layout[current_group_idx]), BLOCK_M);
                const uint32_t current_m_block_cumsum = current_m_cumsum + num_m_blocks;
                if (next_block_idx < current_m_block_cumsum * num_n_blocks) {
                    get_swizzled_block_idx(next_block_idx - current_m_cumsum * num_n_blocks, m_block_idx, n_block_idx);
                    return true;
                }
                current_group_idx++;
                current_m_cumsum = current_m_block_cumsum;
            }
        } else if (GEMM_TYPE == 1) {        // MGroupedContiguous
            if (next_block_idx >= num_blocks)
                return false;
            is_peer_cta_alive = num_n_blocks % MULTICAST == 0 ||
                                num_m_blocks % MULTICAST == 0 ||
                                (next_block_idx ^ 1) < num_blocks;
            get_swizzled_block_idx(next_block_idx, m_block_idx, n_block_idx);
            return true;
        } else if (GEMM_TYPE == 3) {        // Batched
            if (next_block_idx >= num_blocks * NUM_GROUPS)
                return false;
            current_group_idx = next_block_idx / num_blocks;
            const uint32_t block_idx = next_block_idx - current_group_idx * num_blocks;
            if (IS_MULTICAST_ON_A) {
                m_block_idx = block_idx / num_n_blocks;
                n_block_idx = block_idx % num_n_blocks;
            } else {
                m_block_idx = block_idx % num_m_blocks;
                n_block_idx = block_idx / num_m_blocks;
            }
            return true;
        } else {                            // Normal
            if (next_block_idx >= num_blocks)
                return false;
            is_peer_cta_alive = num_n_blocks % MULTICAST == 0 ||
                                num_m_blocks % MULTICAST == 0 ||
                                (next_block_idx ^ 1) < num_blocks;
            get_swizzled_block_idx(next_block_idx, m_block_idx, n_block_idx);
            return true;
        }
    };

    auto is_tma_multicast_valid = [&](uint32_t m_block_idx) -> bool {
        if (num_blocks_in_group == 1)
            return false;
        if (GEMM_TYPE == 0 || GEMM_TYPE == 2 || GEMM_TYPE == 3)
            return true;
        // MGroupedContiguous
        if (IS_MULTICAST_ON_A)
            return true;
        const int group_idx = grouped_layout[m_block_idx * BLOCK_M];
        const int peer_group_idx = grouped_layout[(m_block_idx ^ 1) * BLOCK_M];
        return group_idx == peer_group_idx;
    };

    auto is_computation_valid = [&](uint32_t m_block_idx, uint32_t m_offset) -> bool {
        if (GEMM_TYPE == 0 || GEMM_TYPE == 3)
            return true;
        if (GEMM_TYPE == 1)
            return grouped_layout[m_offset + m_block_idx * BLOCK_M] >= 0;
        // MGroupedMasked
        return m_offset + m_block_idx * BLOCK_M < static_cast<uint32_t>(grouped_layout[current_group_idx]);
    };

    const uint32_t num_total_k_blocks = dg_ceil_div(shape_k, BLOCK_K);

    // ---------------------------------------------------------- pipeline
    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++k_block_idx;
        stage_idx = (stage_idx == NUM_STAGES - 1) ? 0 : stage_idx + 1;
        phase ^= (stage_idx == 0);
    };

    uint32_t m_block_idx, n_block_idx;

    if (warp_idx >= NUM_MATH_THREADS / 32) {
        // ================= TMA (producer) warp-group =====================
        dg_set_max_nreg_dec<40>();

        // Use the third warp of the producer group (warp 0/1 may still be
        // finishing WGMMA when BLOCK_M == 32, upstream detail).
        if (warp_idx == NUM_MATH_THREADS / 32 + 2 && dg_elect_one()) {
            while (get_next_block(m_block_idx, n_block_idx)) {
                const bool is_mc_valid = is_tma_multicast_valid(m_block_idx);
                const uint32_t mc_a = (IS_MULTICAST_ON_A && is_mc_valid) ? MULTICAST : 1;
                const uint32_t mc_b = (!IS_MULTICAST_ON_A && is_mc_valid) ? MULTICAST : 1;

                for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                    // Wait for consumers to release the stage
                    dg_mbarrier_wait(empty_barrier(stage_idx), phase ^ 1);

                    void* bar = full_barrier(stage_idx);
                    const uint32_t k_idx = k_block_idx * BLOCK_K;
                    const uint32_t batch_idx = IS_3D_TMA ? current_group_idx : 0;

                    // ---- A tile ----
                    const bool with_group_offset_a = (GEMM_TYPE == 2);
                    const uint32_t a_outer = get_global_idx(with_group_offset_a, 0, shape_m, BLOCK_M, m_block_idx, 0);
                    if (mc_a == 1) {
                        if (IS_3D_TMA)
                            dg_tma_load_3d(&tensor_map_a, bar, smem_a(stage_idx), k_idx, a_outer, batch_idx);
                        else
                            dg_tma_load_2d(&tensor_map_a, bar, smem_a(stage_idx), k_idx, a_outer);
                    } else {
                        if (IS_3D_TMA)
                            dg_tma_load_3d_multicast(&tensor_map_a, bar, smem_a(stage_idx), k_idx, a_outer, batch_idx, (1u << mc_a) - 1);
                        else
                            dg_tma_load_2d_multicast(&tensor_map_a, bar, smem_a(stage_idx), k_idx, a_outer, (1u << mc_a) - 1);
                    }

#if !IS_BF16
                    // ---- A scale factors ----
                    const uint32_t sf_k_idx = get_global_idx(with_group_offset_a, 2, shape_k_scales, 1, k_block_idx, 0);
                    if (mc_a == 1) {
                        dg_tma_load_2d(&tensor_map_sfa, bar, smem_sfa(stage_idx), m_block_idx * BLOCK_M, sf_k_idx);
                    } else {
                        dg_tma_load_2d_multicast(&tensor_map_sfa, bar, smem_sfa(stage_idx), m_block_idx * BLOCK_M, sf_k_idx, (1u << mc_a) - 1);
                    }
#endif

                    // ---- B tile ----
                    const uint32_t b_outer = get_global_idx(true, 0, shape_n, BLOCK_N, n_block_idx, m_block_idx);
                    if (mc_b == 1) {
                        if (IS_3D_TMA)
                            dg_tma_load_3d(&tensor_map_b, bar, smem_b(stage_idx), k_idx, b_outer, batch_idx);
                        else
                            dg_tma_load_2d(&tensor_map_b, bar, smem_b(stage_idx), k_idx, b_outer);
                    } else {
                        if (IS_3D_TMA)
                            dg_tma_load_3d_multicast(&tensor_map_b, bar, smem_b(stage_idx), k_idx, b_outer, batch_idx, (1u << mc_b) - 1);
                        else
                            dg_tma_load_2d_multicast(&tensor_map_b, bar, smem_b(stage_idx), k_idx, b_outer, (1u << mc_b) - 1);
                    }

                    dg_mbarrier_arrive_expect_tx(bar, SMEM_A_PER_STAGE + SMEM_B_PER_STAGE
#if !IS_BF16
                                                        + SMEM_SFA_PER_STAGE
#endif
                    );
                }
            }
            // Drain: wait all stages released before destroying cluster barriers
            if (MULTICAST > 1) {
                for (uint32_t i = 0; i < NUM_STAGES; advance_pipeline(i))
                    dg_mbarrier_wait(empty_barrier(stage_idx), phase ^ 1);
            }
        }
    } else {
        // ================= math (consumer) warp-groups ====================
        dg_set_max_nreg_inc<NUM_MATH_THREADS == 128 ? 248 : 232>();

        const uint32_t math_wg_idx = threadIdx.x / 128;
        const uint32_t r_0 = warp_idx * 16 + lane_idx / 4;
        const uint32_t r_1 = r_0 + 8;

        // Precompute WGMMA smem descriptors at stage-0 base; low 32 bits are
        // then offset by stage/m/k (upstream optimization).
        const uint64_t a_desc_base = dg_make_smem_desc(smem_a(0) + math_wg_idx * WGMMA_M * BLOCK_K, 1, 0, 1024);
        const uint64_t b_desc_base = dg_make_smem_desc(smem_b(0), 1, 0, 1024);
        const uint32_t a_desc_lo = static_cast<uint32_t>(a_desc_base);
        const uint32_t b_desc_lo = static_cast<uint32_t>(b_desc_base);

        const bool do_wgmma_store = BLOCK_M >= WGMMA_M || warp_idx < NUM_WGMMA_STORE_THREADS / 32;

        while (get_next_block(m_block_idx, n_block_idx)) {
#if !IS_BF16
            // ---------------- stage B scales into smem (math warps) --------
            // All math warps except warp 0 (which may still be in WGMMA).
            uint32_t num_former_iters = BLOCK_N / 8, num_full_iters = num_former_iters;
            #if !MUST_USE_UNIFORMED_SCALE_B
            num_former_iters = dg_min_u32(BLOCK_N, BLOCK_K - (n_block_idx * BLOCK_N) % BLOCK_K) / 8;
            num_full_iters = dg_min_u32(shape_n - n_block_idx * BLOCK_N, BLOCK_N) / 8;
            #endif
            const uint32_t num_sfb = shape_k_scales * (num_former_iters >= num_full_iters ? 1 : 2);

            if (threadIdx.x >= 32) {
                const uint32_t previous_group_offset =
                    get_global_idx(true, 2, shape_n_sfb * shape_k_scales, 0, 0, m_block_idx);
                // K-major sfb layout: (groups*n_tiles, k_blocks), k contiguous
                const uint32_t stride_k_sfb = 1;
                const uint32_t stride_n_sfb = shape_k_scales;
                const float* local_sfb = sfb + previous_group_offset +
                                         ((n_block_idx * BLOCK_N) / BLOCK_K) * stride_n_sfb;
                for (uint32_t i = threadIdx.x - 32; i < num_sfb; i += NUM_MATH_THREADS - 32)
                    dg_st_shared_f32(smem_sfb + i,
                                     i < shape_k_scales ? local_sfb[i * stride_k_sfb]
                                                        : local_sfb[(i - shape_k_scales) * stride_k_sfb + stride_n_sfb]);
            }
            dg_named_barrier_sync(0, NUM_MATH_THREADS);
#endif

#if !IS_BF16
            float accum[NUM_ACCUM];
            #pragma unroll
            for (uint32_t i = 0; i < NUM_ACCUM; ++i)
                accum[i] = 0.0f;
#endif
            float final_accum[NUM_ACCUM * (BLOCK_M / WAVE_BLOCK_M)];
            #pragma unroll
            for (uint32_t i = 0; i < NUM_ACCUM * (BLOCK_M / WAVE_BLOCK_M); ++i)
                final_accum[i] = 0.0f;

            auto empty_barrier_arrive = [&]() {
                if (MULTICAST == 1) {
                    if (lane_idx == 0)
                        dg_mbarrier_arrive(empty_barrier(stage_idx));
                } else {
                    const uint32_t target_cta = is_peer_cta_alive ? lane_idx : dg_block_rank_in_cluster();
                    if (lane_idx < MULTICAST)
                        dg_mbarrier_arrive_cluster(empty_barrier(stage_idx), target_cta);
                }
            };

            if (is_computation_valid(m_block_idx, math_wg_idx * WGMMA_M)) {
                for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                    const uint32_t a_desc_stage_lo = a_desc_lo + stage_idx * (SMEM_A_PER_STAGE / 16);
                    const uint32_t b_desc_stage_lo = b_desc_lo + stage_idx * (SMEM_B_PER_STAGE / 16);

#if !IS_BF16
                    // Read B scales (k-block shared across the whole N tile)
                    const float scale_b_0 = dg_ld_shared_f32(smem_sfb + k_block_idx);
                    float scale_b_1 = scale_b_0;
                    #if !MUST_USE_UNIFORMED_SCALE_B
                    scale_b_1 = dg_ld_shared_f32(smem_sfb + k_block_idx + shape_k_scales);
                    #endif
#endif

                    // Wait for TMA completion
                    dg_mbarrier_wait(full_barrier(stage_idx), phase);

                    #pragma unroll
                    for (uint32_t local_idx = 0; local_idx < BLOCK_M / WAVE_BLOCK_M; ++local_idx) {
                        const uint32_t m_offset = local_idx * WAVE_BLOCK_M;

#if !IS_BF16
                        // Read A scales BEFORE wgmma arrive (upstream: avoid
                        // next-scheduled block polluting results)
                        const float scale_a_0 = do_wgmma_store ? dg_ld_shared_f32(smem_sfa(stage_idx) + r_0 + m_offset) : 0.0f;
                        const float scale_a_1 = do_wgmma_store ? dg_ld_shared_f32(smem_sfa(stage_idx) + r_1 + m_offset) : 0.0f;

                        // ---- WGMMA over the k-block, accumulate into `accum` ----
                        #pragma unroll
                        for (uint32_t i = 0; i < NUM_ACCUM; ++i)
                            dg_warpgroup_fence(accum[i]);
                        dg_warpgroup_arrive();
                        #pragma unroll
                        for (uint32_t k = 0; k < NUM_WGMMA_PER_K; ++k) {
                            const uint64_t desc_a = a_desc_stage_lo +
                                ((m_offset * BLOCK_K + k * WGMMA_K) * AB_ELEM_SIZE) / 16;
                            const uint64_t desc_b = b_desc_stage_lo + (k * WGMMA_K * AB_ELEM_SIZE) / 16;
                            if (k == 0)
                                WGMMA_FN<true>(accum, desc_a, desc_b);
                            else
                                WGMMA_FN<false>(accum, desc_a, desc_b);
                        }
                        dg_warpgroup_commit_batch();
                        #pragma unroll
                        for (uint32_t i = 0; i < NUM_ACCUM; ++i)
                            dg_warpgroup_fence(accum[i]);
                        dg_warpgroup_wait<0>();

                        if (local_idx == BLOCK_M / WAVE_BLOCK_M - 1)
                            empty_barrier_arrive();

                        if (!do_wgmma_store)
                            continue;

                        // ---- promotion with scale factors ----
                        const float scale_0_0 = scale_a_0 * scale_b_0;
                        const float scale_1_0 = scale_a_1 * scale_b_0;
                        float scale_0_1 = scale_0_0, scale_1_1 = scale_1_0;
                        #if !MUST_USE_UNIFORMED_SCALE_B
                        scale_0_1 = scale_a_0 * scale_b_1;
                        scale_1_1 = scale_a_1 * scale_b_1;
                        #endif
                        float* shifted = final_accum + NUM_ACCUM * local_idx;
                        #pragma unroll
                        for (uint32_t i = 0; i < NUM_ACCUM / 4; ++i) {
                            const bool pred = MUST_USE_UNIFORMED_SCALE_B || i < num_former_iters;
                            shifted[i * 4 + 0] += (pred ? scale_0_0 : scale_0_1) * accum[i * 4 + 0];
                            shifted[i * 4 + 1] += (pred ? scale_0_0 : scale_0_1) * accum[i * 4 + 1];
                            shifted[i * 4 + 2] += (pred ? scale_1_0 : scale_1_1) * accum[i * 4 + 2];
                            shifted[i * 4 + 3] += (pred ? scale_1_0 : scale_1_1) * accum[i * 4 + 3];
                        }
#else
                        // ---- bf16: accumulate directly into final_accum ----
                        float* target = final_accum + NUM_ACCUM * local_idx;
                        #pragma unroll
                        for (uint32_t i = 0; i < NUM_ACCUM; ++i)
                            dg_warpgroup_fence(target[i]);
                        dg_warpgroup_arrive();
                        #pragma unroll
                        for (uint32_t k = 0; k < NUM_WGMMA_PER_K; ++k) {
                            const uint64_t desc_a = a_desc_stage_lo +
                                ((m_offset * BLOCK_K + k * WGMMA_K) * AB_ELEM_SIZE) / 16;
                            const uint64_t desc_b = b_desc_stage_lo + (k * WGMMA_K * AB_ELEM_SIZE) / 16;
                            if (k == 0 && k_block_idx == 0)
                                WGMMA_FN<true>(target, desc_a, desc_b);
                            else
                                WGMMA_FN<false>(target, desc_a, desc_b);
                        }
                        dg_warpgroup_commit_batch();
                        #pragma unroll
                        for (uint32_t i = 0; i < NUM_ACCUM; ++i)
                            dg_warpgroup_fence(target[i]);
                        dg_warpgroup_wait<0>();

                        if (local_idx == BLOCK_M / WAVE_BLOCK_M - 1)
                            empty_barrier_arrive();
#endif
                    }
                }
            } else {
                // Skip compute, but keep the pipeline moving
                for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                    dg_mbarrier_wait(full_barrier(stage_idx), phase);
                    empty_barrier_arrive();
                }
            }

            if (!do_wgmma_store)
                continue;

            // ---------------- epilogue: STSM + TMA store -------------------
            if (threadIdx.x < BLOCK_N / TMA_D_BLOCK_N)
                dg_tma_store_wait<0>();
            dg_named_barrier_sync(1, NUM_WGMMA_STORE_THREADS);

            #pragma unroll
            for (uint32_t local_idx = 0; local_idx < BLOCK_M / WAVE_BLOCK_M; ++local_idx) {
                const uint32_t m_offset = local_idx * WAVE_BLOCK_M;
                const float* shifted = final_accum + NUM_ACCUM * local_idx;
                #pragma unroll
                for (uint32_t i = 0; i < NUM_ACCUM / 4; ++i) {
                    uint8_t* smem_ptr;
                    if (SWIZZLE_D > 0) {
                        // Swizzled SMEM D atom: (BLOCK_M rows x SWIZZLE_D bytes)
                        constexpr uint32_t kNumBankGroupBytes = 16;
                        const uint32_t atom_offset = i / (TMA_D_BLOCK_N / 8);
                        const uint32_t in_atom_offset = i % (TMA_D_BLOCK_N / 8);
                        const uint32_t bank_group_index = in_atom_offset + lane_idx * (SWIZZLE_D / kNumBankGroupBytes);
                        constexpr bool kHasShortcut = (SWIZZLE_D / kNumBankGroupBytes) == 8;
                        const uint32_t row = kHasShortcut ? (in_atom_offset + lane_idx) : (bank_group_index / 8);
                        uint32_t col = kHasShortcut ? in_atom_offset : (bank_group_index % 8);
                        col ^= row % (SWIZZLE_D / 16);

                        smem_ptr = reinterpret_cast<uint8_t*>(smem_d) +
                                   warp_idx * (16 * SWIZZLE_D) +
                                   m_offset * SWIZZLE_D +
                                   atom_offset * BLOCK_M * SWIZZLE_D +
                                   row * (kNumBankGroupBytes * 8) + col * kNumBankGroupBytes;
                    } else {
                        smem_ptr = reinterpret_cast<uint8_t*>(smem_d +
                                   (m_offset + warp_idx * 16 + lane_idx) * BLOCK_N + i * 8);
                    }
                    const uint32_t v0 = PACK_FN(shifted[i * 4 + 0], shifted[i * 4 + 1]);
                    const uint32_t v1 = PACK_FN(shifted[i * 4 + 2], shifted[i * 4 + 3]);
                    dg_stsm_x2(v0, v1, smem_ptr);
                }
            }
            dg_tma_store_fence();
            dg_named_barrier_sync(1, NUM_WGMMA_STORE_THREADS);

            // TMA store back to global
            const bool with_group_offset_d = (GEMM_TYPE == 2);
            if (threadIdx.x < BLOCK_N / TMA_D_BLOCK_N) {
                const uint32_t in_block_n_offset = threadIdx.x * TMA_D_BLOCK_N;
                const uint16_t* smem_ptr = smem_d + in_block_n_offset * BLOCK_M;
                const uint32_t n_idx = n_block_idx * BLOCK_N + in_block_n_offset;
                const uint32_t m_idx = get_global_idx(with_group_offset_d, 0, shape_m, BLOCK_M, m_block_idx, 0);
                if (IS_3D_TMA) {
                    dg_tma_store_3d(&tensor_map_d, smem_ptr, n_idx, m_idx, current_group_idx);
                } else {
                    dg_tma_store_2d(&tensor_map_d, smem_ptr, n_idx, m_idx);
                }
                dg_tma_store_commit_group();
            }
            __syncwarp();
        }
    }
#endif // __CUDA_ARCH__ >= 900
}

// Number of groups is passed via a compile-time constant for the masked /
// batched / contiguous schedulers (upstream template parameter).
"#;

/// Helper macros needed by the static asserts (NVRTC has no constexpr lib).
const HELPERS: &str = r#"
template <uint32_t A, uint32_t B> struct CeilDiv { static const uint32_t value = (A + B - 1) / B; };
#define CEIL_DIV_CONST(a, b) ((a + b - 1) / b)
DG_INLINE constexpr uint32_t const_gcd(uint32_t a, uint32_t b) { return b == 0 ? a : const_gcd(b, a % b); }
#define CONST_GCD(a, b) (dg_const_gcd_impl(a, b))
__device__ constexpr uint32_t dg_const_gcd_impl(uint32_t a, uint32_t b) { return b == 0 ? a : dg_const_gcd_impl(b, a % b); }
"#;

/// Build the full CUDA TU for one GEMM configuration.
#[allow(clippy::too_many_arguments)]
pub fn build_gemm_kernel_source(
    block_m: u32,
    block_n: u32,
    block_k: u32,
    num_stages: u32,
    swizzle_a: u32,
    swizzle_b: u32,
    swizzle_d: u32,
    num_tma_threads: u32,
    num_math_threads: u32,
    cluster_size: u32,
    is_multicast_on_a: bool,
    num_sms: u32,
    gemm_type: u32, // 0=Normal 1=MGroupedContiguous 2=MGroupedMasked 3=Batched
    num_groups: u32,
    is_bf16: bool,
) -> String {
    let wgmma_fn = if is_bf16 {
        format!(
            "#define WGMMA_FN dg_wgmma_bf16\n{}",
            gen_wgmma_bf16(block_n)
        )
    } else {
        format!("#define WGMMA_FN dg_wgmma_fp8\n{}", gen_wgmma_fp8(block_n))
    };
    // The epilogue pack is bf16 for both fp8 and bf16 kernels (output dtype).
    let pack_fn = "#define PACK_FN(lo, hi) dg_pack_bf16x2(lo, hi)";

    let src = super::subst(
        KERNEL_TEMPLATE,
        &[
            ("BLOCK_M", &block_m.to_string()),
            ("BLOCK_N", &block_n.to_string()),
            ("BLOCK_K", &block_k.to_string()),
            ("NUM_STAGES", &num_stages.to_string()),
            ("SWIZZLE_A", &swizzle_a.to_string()),
            ("SWIZZLE_B", &swizzle_b.to_string()),
            ("SWIZZLE_D", &swizzle_d.to_string()),
            ("NUM_TMA_THREADS", &num_tma_threads.to_string()),
            ("NUM_MATH_THREADS", &num_math_threads.to_string()),
            ("MULTICAST", &cluster_size.to_string()),
            (
                "IS_MULTICAST_ON_A",
                if is_multicast_on_a { "1" } else { "0" },
            ),
            ("NUM_SMS", &num_sms.to_string()),
            ("GEMM_TYPE", &gemm_type.to_string()),
            ("IS_BF16", if is_bf16 { "1" } else { "0" }),
        ],
    );

    // NUM_GROUPS is referenced by the scheduler; define per-TU.
    let groups_def = format!("#define NUM_GROUPS {num_groups}u\n");

    format!("{COMMON_HEADER}\n{HELPERS}\n{groups_def}\n{wgmma_fn}\n{pack_fn}\n{src}")
}
