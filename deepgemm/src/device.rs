//! Device context: owns the `cudarc` device, exposes arch capabilities and
//! the JIT engine bound to this device.

use std::sync::Arc;

use cudarc::driver::safe::{CudaContext, CudaStream};

use crate::jit::JitEngine;
use crate::types::DgResult;

/// Capabilities of the compute capability we compile for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Arch {
    pub major: u32,
    pub minor: u32,
    /// e.g. `sm_90a` (architecture-specific code, needed for WGMMA/TMA).
    pub nvrtc_arch: &'static str,
    /// Max dynamic shared memory per block (bytes), excluding the reserved 1KB.
    pub smem_capacity: u32,
    pub num_sms: u32,
}

impl Arch {
    pub fn is_hopper(&self) -> bool {
        self.major == 9
    }
    pub fn is_blackwell(&self) -> bool {
        self.major == 10 || self.major == 12
    }
    pub fn is_ada(&self) -> bool {
        self.major == 8 && self.minor == 9
    }
    /// Warp-group MMA (wgmma) available: Hopper and above.
    pub fn has_wgmma(&self) -> bool {
        self.major >= 9
    }
}

/// Shared context: device + stream + JIT cache.
pub struct DgContext {
    pub device: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    pub arch: Arch,
    pub jit: JitEngine,
}

impl DgContext {
    /// Create a context on the given device ordinal (0 by default).
    pub fn new(ordinal: usize) -> DgResult<Self> {
        let device = CudaContext::new(ordinal)
            .map_err(|e| crate::types::DgError::Driver(format!("{e:?}")))?;
        Self::from_device(device)
    }

    pub fn from_device(device: Arc<CudaContext>) -> DgResult<Self> {
        let stream = device.default_stream();
        let (major, minor) = device
            .attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .and_then(|ma| {
                device
                    .attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
                    .map(|mi| (ma as u32, mi as u32))
            })
            .map_err(|e| crate::types::DgError::Driver(format!("{e:?}")))?;

        let num_sms = device
            .attribute(
                cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            )
            .map_err(|e| crate::types::DgError::Driver(format!("{e:?}")))?
            as u32;

        // Max dynamic smem per block (query is the total incl. reserved 1KB).
        let max_smem = device
            .attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)
            .map_err(|e| crate::types::DgError::Driver(format!("{e:?}")))? as u32;
        // Upstream DeepGEMM uses 232448 (= 227KB) on H100; use the queried value.
        let smem_capacity = max_smem.max(48 * 1024) - 1024;

        let arch = if major >= 9 {
            Arch {
                major,
                minor,
                nvrtc_arch: "sm_90a",
                smem_capacity,
                num_sms,
            }
        } else if major == 8 && minor == 9 {
            Arch {
                major,
                minor,
                nvrtc_arch: "sm_89",
                smem_capacity,
                num_sms,
            }
        } else if major >= 10 {
            Arch {
                major,
                minor,
                nvrtc_arch: "sm_100a",
                smem_capacity,
                num_sms,
            }
        } else {
            Arch {
                major,
                minor,
                nvrtc_arch: "sm_80",
                smem_capacity,
                num_sms,
            }
        };

        let jit = JitEngine::new(&arch)?;
        Ok(Self {
            device,
            stream,
            arch,
            jit,
        })
    }

    /// Number of SMs to fill with the persistent kernel.
    pub fn num_sms(&self) -> u32 {
        self.arch.num_sms
    }

    /// Synchronize the stream.
    pub fn sync(&self) -> DgResult<()> {
        self.device
            .synchronize()
            .map_err(|e| crate::types::DgError::Driver(format!("{e:?}")))
    }
}

impl std::fmt::Debug for DgContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DgContext")
            .field(
                "arch",
                &format!("sm_{}{}", self.arch.major, self.arch.minor),
            )
            .field("num_sms", &self.arch.num_sms)
            .finish()
    }
}
