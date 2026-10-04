//! Shared, self-contained CUDA "header" prepended to every kernel TU.
//!
//! Faithful PTX-level ports of the cutlass/cute primitives used by upstream
//! DeepGEMM (mbarrier pipeline, TMA tensor copies, WGMMA, STSM, named
//! barriers, register reconfiguration, warp election) — with zero external
//! includes so NVRTC can compile it directly.

pub const COMMON_HEADER: &str = r#"
// ============================================================================
// deepgemm-rust common device header (NVRTC, self-contained)
// ============================================================================
typedef unsigned int       uint32_t;
typedef int                int32_t;
typedef unsigned long long uint64_t;
typedef long long          int64_t;
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;

#define DG_INLINE __device__ __forceinline__

// 128-byte TMA descriptor passed by value as a kernel parameter.
struct alignas(64) TmaDescriptor { uint64_t data[8]; };

DG_INLINE uint32_t smem_u32(const void* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

DG_INLINE uint32_t dg_lane_idx() { return threadIdx.x % 32; }
DG_INLINE uint32_t dg_warp_idx() { return threadIdx.x / 32; }

// Elect a single lane of the warp (sm_90+).
DG_INLINE bool dg_elect_one() {
    uint32_t pred = 0;
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "elect.sync _|P, 0xffffffff;\n"
        "selp.u32 %0, 1, 0, P;\n"
        "}\n"
        : "=r"(pred));
    return pred != 0;
}

// ------------------------------- mbarrier ----------------------------------
// A transaction barrier: producer arrives with an expected byte count,
// consumers wait on phase parity.
DG_INLINE void dg_mbarrier_init(void* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(bar)), "r"(count));
}

DG_INLINE void dg_fence_barrier_init() {
    // NOTE: PTX only defines the .cluster scope for this fence; it is valid
    // (and required) even for single-CTA launches (upstream cutlass usage).
    asm volatile("fence.mbarrier_init.release.cluster;");
}

DG_INLINE void dg_mbarrier_arrive_expect_tx(void* bar, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                 :: "r"(smem_u32(bar)), "r"(tx_bytes));
}

DG_INLINE void dg_mbarrier_arrive(void* bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(bar)));
}

// Arrive at a barrier living in another CTA of the cluster.
DG_INLINE void dg_mbarrier_arrive_cluster(void* bar, uint32_t target_cta) {
    uint32_t addr = smem_u32(bar);
    uint32_t remote;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote) : "r"(addr), "r"(target_cta));
    asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];" :: "r"(remote));
}

DG_INLINE void dg_mbarrier_wait(void* bar, uint32_t phase) {
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "LAB_WAIT:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1, 100000000;\n"
        "@P bra DONE;\n"
        "bra LAB_WAIT;\n"
        "DONE:\n"
        "}\n"
        :: "r"(smem_u32(bar)), "r"(phase));
}

// ------------------------------ cluster ------------------------------------
DG_INLINE uint32_t dg_block_rank_in_cluster() {
    uint32_t rank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
    return rank;
}

DG_INLINE void dg_cluster_sync_relaxed() {
    asm volatile("barrier.cluster.arrive.relaxed;");
    asm volatile("barrier.cluster.wait;");
}

// --------------------------- named barriers --------------------------------
DG_INLINE void dg_named_barrier_sync(uint32_t id, uint32_t count) {
    asm volatile("barrier.sync.aligned %0, %1;" :: "r"(id), "r"(count));
}

// ------------------------ register reallocation ----------------------------
template <int N>
DG_INLINE void dg_set_max_nreg_dec() {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" :: "n"(N < 24 ? 24 : N));
}
template <int N>
DG_INLINE void dg_set_max_nreg_inc() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(N));
}

// ------------------------------ WGMMA fences -------------------------------
DG_INLINE void dg_warpgroup_arrive() {
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
}
DG_INLINE void dg_warpgroup_commit_batch() {
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
}
template <int N>
DG_INLINE void dg_warpgroup_wait() {
    asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N) : "memory");
}
DG_INLINE void dg_warpgroup_fence(float& reg) {
    asm volatile("" : "+f"(reg) :: "memory");
}

// ------------------------------ TMA loads ----------------------------------
// 2-D tiled TMA load into shared memory with mbarrier completion.
DG_INLINE void dg_tma_load_2d(const TmaDescriptor* desc, void* bar,
                              void* smem_dst, uint32_t c0, uint32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%3, %4}], [%2];"
        :: "r"(smem_u32(smem_dst)), "l"(reinterpret_cast<uint64_t>(desc)),
           "r"(smem_u32(bar)), "r"(c0), "r"(c1)
        : "memory");
}

// 2-D multicast TMA load (replicates to the same smem/barrier offsets in all
// CTAs of `cta_mask`).
DG_INLINE void dg_tma_load_2d_multicast(const TmaDescriptor* desc, void* bar,
                                        void* smem_dst, uint32_t c0, uint32_t c1,
                                        uint16_t cta_mask) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
        ".multicast::cluster [%0], [%1, {%3, %4}], [%2], %5;"
        :: "r"(smem_u32(smem_dst)), "l"(reinterpret_cast<uint64_t>(desc)),
           "r"(smem_u32(bar)), "r"(c0), "r"(c1), "h"(cta_mask)
        : "memory");
}

// 3-D tiled TMA load (batched tensors).
DG_INLINE void dg_tma_load_3d(const TmaDescriptor* desc, void* bar,
                              void* smem_dst, uint32_t c0, uint32_t c1, uint32_t c2) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%3, %4, %5}], [%2];"
        :: "r"(smem_u32(smem_dst)), "l"(reinterpret_cast<uint64_t>(desc)),
           "r"(smem_u32(bar)), "r"(c0), "r"(c1), "r"(c2)
        : "memory");
}

DG_INLINE void dg_tma_load_3d_multicast(const TmaDescriptor* desc, void* bar,
                                        void* smem_dst, uint32_t c0, uint32_t c1,
                                        uint32_t c2, uint16_t cta_mask) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
        ".multicast::cluster [%0], [%1, {%3, %4, %5}], [%2], %6;"
        :: "r"(smem_u32(smem_dst)), "l"(reinterpret_cast<uint64_t>(desc)),
           "r"(smem_u32(bar)), "r"(c0), "r"(c1), "r"(c2), "h"(cta_mask)
        : "memory");
}

// ------------------------------ TMA stores ---------------------------------
DG_INLINE void dg_tma_store_2d(const TmaDescriptor* desc, const void* smem_src,
                               uint32_t c0, uint32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group"
        " [%0, {%2, %3}], [%1];"
        :: "l"(reinterpret_cast<uint64_t>(desc)), "r"(smem_u32(smem_src)),
           "r"(c0), "r"(c1)
        : "memory");
}

DG_INLINE void dg_tma_store_3d(const TmaDescriptor* desc, const void* smem_src,
                               uint32_t c0, uint32_t c1, uint32_t c2) {
    asm volatile(
        "cp.async.bulk.tensor.3d.global.shared::cta.bulk_group"
        " [%0, {%2, %3, %4}], [%1];"
        :: "l"(reinterpret_cast<uint64_t>(desc)), "r"(smem_u32(smem_src)),
           "r"(c0), "r"(c1), "r"(c2)
        : "memory");
}

DG_INLINE void dg_tma_store_commit_group() {
    asm volatile("cp.async.bulk.commit_group;");
}

template <int N>
DG_INLINE void dg_tma_store_wait() {
    asm volatile("cp.async.bulk.wait_group %0;" :: "n"(N) : "memory");
}

DG_INLINE void dg_tma_store_fence() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

DG_INLINE void dg_prefetch_tma_descriptor(const TmaDescriptor* desc) {
    asm volatile("prefetch.tensormap [%0];" :: "l"(reinterpret_cast<uint64_t>(desc)));
}

// --------------------------- shared memory I/O ------------------------------
DG_INLINE float dg_ld_shared_f32(const float* ptr) {
    return *ptr;
}
DG_INLINE void dg_st_shared_f32(float* ptr, float v) {
    *ptr = v;
}

// stmatrix.x2: two 8x8 b16 matrices; lanes 0-15 provide row addresses.
DG_INLINE void dg_stsm_x2(uint32_t r0, uint32_t r1, void* smem_ptr) {
    asm volatile(
        "stmatrix.sync.aligned.x2.m8n8.shared.b16 [%0], {%1, %2};"
        :: "r"(smem_u32(smem_ptr)), "r"(r0), "r"(r1));
}

// Pack two floats into one bf16x2 u32.
DG_INLINE uint32_t dg_pack_bf16x2(float lo, float hi) {
    uint32_t r;
    asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(r) : "f"(lo), "f"(hi));
    return r;
}

// Pack two floats into one f16x2 u32.
DG_INLINE uint32_t dg_pack_f16x2(float lo, float hi) {
    uint32_t r;
    asm volatile("cvt.rn.f16x2.f32 %0, %2, %1;" : "=r"(r) : "f"(lo), "f"(hi));
    return r;
}

// ------------------------------ grid dependency ----------------------------
DG_INLINE void dg_grid_dependency_sync() {
    asm volatile("griddepcontrol.wait;" ::: "memory");
}

// --------------------------- WGMMA smem descriptor --------------------------
// GmmaDescriptor: 64-bit {addr[0:14)<<4, LBO>>4 <<16, SBO>>4 <<32, layout<<62}
// layout_type: 0 = interleaved, 1 = B128 swizzle, 2 = B64, 3 = B32.
DG_INLINE uint64_t dg_make_smem_desc(const void* smem_ptr, uint32_t layout_type,
                                     uint32_t leading_byte_offset,
                                     uint32_t stride_byte_offset) {
    uint64_t desc = 0;
    desc |= static_cast<uint64_t>(smem_u32(smem_ptr) >> 4);
    desc |= static_cast<uint64_t>(leading_byte_offset >> 4) << 16;
    desc |= static_cast<uint64_t>(stride_byte_offset >> 4) << 32;
    desc |= static_cast<uint64_t>(layout_type) << 62;
    return desc;
}

// ------------------------------- utilities ---------------------------------
DG_INLINE constexpr uint32_t dg_ceil_div(uint32_t a, uint32_t b) { return (a + b - 1) / b; }
DG_INLINE uint32_t dg_min_u32(uint32_t a, uint32_t b) { return a < b ? a : b; }
DG_INLINE int32_t  dg_max_i32(int32_t a, int32_t b) { return a > b ? a : b; }

DG_INLINE void dg_debug_barrier() { __syncthreads(); }
"#;
