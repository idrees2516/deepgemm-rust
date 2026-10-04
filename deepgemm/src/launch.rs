//! Low-level kernel launch via `cuLaunchKernelEx`.
//!
//! Needed for features the simple cudarc launch path doesn't expose:
//! * thread-block clusters (TMA multicast),
//! * > 48 KB dynamic shared memory (opt-in via `cuFuncSetAttribute`),
//! * passing 128-byte `CUtensorMap` descriptors **by value**.

use std::sync::Arc;

use cudarc::driver::safe::{CudaFunction, CudaStream};
use cudarc::driver::sys::{
    self, cuFuncSetAttribute, cuLaunchKernelEx, CUfunction_attribute, CUlaunchAttribute,
    CUlaunchAttributeID, CUlaunchConfig,
};

use crate::types::{DgError, DgResult};

/// A kernel argument that can be serialized into the driver parameter buffer.
pub trait KernelArg {
    fn write_to(&self, buf: &mut Vec<u8>);
}

impl KernelArg for u32 {
    fn write_to(&self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&self.to_ne_bytes());
    }
}
impl KernelArg for i32 {
    fn write_to(&self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&self.to_ne_bytes());
    }
}
impl KernelArg for u64 {
    fn write_to(&self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&self.to_ne_bytes());
    }
}
impl KernelArg for f32 {
    fn write_to(&self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&self.to_ne_bytes());
    }
}
/// Raw device pointer (CUdeviceptr = u64).
#[derive(Clone, Copy)]
pub struct DevPtr(pub u64);
impl KernelArg for DevPtr {
    fn write_to(&self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&self.0.to_ne_bytes());
    }
}
impl<T> KernelArg for *const T {
    fn write_to(&self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&(*self as u64).to_ne_bytes());
    }
}
impl<T> KernelArg for *mut T {
    fn write_to(&self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&(*self as u64).to_ne_bytes());
    }
}
/// `CUtensorMap` passed by value (128 bytes; aligned to 64 by the builder).
impl KernelArg for sys::CUtensorMap {
    fn write_to(&self, buf: &mut Vec<u8>) {
        let bytes = unsafe {
            std::slice::from_raw_parts(
                self as *const Self as *const u8,
                std::mem::size_of::<sys::CUtensorMap>(),
            )
        };
        buf.extend_from_slice(bytes);
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct LaunchGrid {
    pub grid_x: u32,
    pub grid_y: u32,
    pub grid_z: u32,
    pub block_x: u32,
    pub block_y: u32,
    pub block_z: u32,
    /// Dynamic shared memory bytes.
    pub smem: u32,
    /// Cluster (x, y, z); `None` or (1, 1, 1) = no cluster.
    pub cluster: Option<(u32, u32, u32)>,
}

impl LaunchGrid {
    pub fn new(grid_x: u32, block_x: u32, smem: u32) -> Self {
        Self {
            grid_x,
            block_x,
            smem,
            ..Default::default()
        }
    }
}

fn check(res: sys::CUresult) -> DgResult<()> {
    if res == sys::CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(DgError::Driver(format!("cuLaunchKernelEx failed: {res:?}")))
    }
}

/// Builder that serializes kernel arguments and launches via `cuLaunchKernelEx`.
pub struct ArgBuilder {
    buf: Vec<u8>,
    ptrs: Vec<(usize, usize)>,
}

impl ArgBuilder {
    pub fn new() -> Self {
        Self {
            buf: Vec::with_capacity(1024),
            ptrs: Vec::new(),
        }
    }

    /// Push an argument, aligning its start offset (64 for `CUtensorMap`).
    pub fn push_aligned(&mut self, arg: &dyn KernelArg, align: usize) -> &mut Self {
        while self.buf.len() % align != 0 {
            self.buf.push(0);
        }
        let start = self.buf.len();
        arg.write_to(&mut self.buf);
        self.ptrs.push((start, self.buf.len() - start));
        self
    }

    pub fn push(&mut self, arg: &dyn KernelArg) -> &mut Self {
        let start = self.buf.len();
        arg.write_to(&mut self.buf);
        self.ptrs.push((start, self.buf.len() - start));
        self
    }

    /// Launch with the collected args.
    ///
    /// # Safety
    /// Args must match the kernel signature.
    pub unsafe fn launch(
        &self,
        func: &CudaFunction,
        stream: &Arc<CudaStream>,
        grid: LaunchGrid,
    ) -> DgResult<()> {
        check(cuFuncSetAttribute(
            func.cu_function(),
            CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            232448 - 1024, // matches upstream DeepGEMM's usable smem budget
        ))?;

        let mut param_ptrs: Vec<*mut std::ffi::c_void> = self
            .ptrs
            .iter()
            .map(|(s, _)| self.buf.as_ptr().add(*s) as *mut _)
            .collect();

        let mut attrs: Vec<CUlaunchAttribute> = Vec::new();
        if let Some((cx, cy, cz)) = grid.cluster {
            if cx > 1 || cy > 1 || cz > 1 {
                let mut attr = std::mem::zeroed::<CUlaunchAttribute>();
                attr.id = CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
                attr.value.clusterDim.x = cx;
                attr.value.clusterDim.y = cy;
                attr.value.clusterDim.z = cz;
                attrs.push(attr);
            }
        }

        let config = CUlaunchConfig {
            gridDimX: grid.grid_x,
            gridDimY: grid.grid_y.max(1),
            gridDimZ: grid.grid_z.max(1),
            blockDimX: grid.block_x,
            blockDimY: grid.block_y.max(1),
            blockDimZ: grid.block_z.max(1),
            sharedMemBytes: grid.smem,
            hStream: stream.cu_stream(),
            attrs: attrs.as_mut_ptr(),
            numAttrs: attrs.len() as u32,
        };

        check(cuLaunchKernelEx(
            &config,
            func.cu_function(),
            param_ptrs.as_mut_ptr(),
            std::ptr::null_mut(),
        ))
    }
}

impl Default for ArgBuilder {
    fn default() -> Self {
        Self::new()
    }
}
