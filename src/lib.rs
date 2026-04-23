#![cfg_attr(not(feature = "gpu-metal"), no_std)]
extern crate alloc;

pub use p3_whir::{constraints, fiat_shamir, parameters, sumcheck};
pub mod whir;

#[cfg(all(feature = "gpu-metal", any(target_os = "macos", target_os = "ios")))]
pub mod gpu_dft;

/// DFT backend for [`p3_baby_bear::BabyBear`]: Metal NTT on macOS/iOS with `gpu-metal`, else CPU radix-2.
#[cfg(all(feature = "gpu-metal", any(target_os = "macos", target_os = "ios")))]
pub use gpu_dft::MetalBabyBearDft as BabyBearDft;
#[cfg(all(feature = "gpu-metal", any(target_os = "macos", target_os = "ios")))]
pub use gpu_dft::GpuChallenger;

#[cfg(not(all(feature = "gpu-metal", any(target_os = "macos", target_os = "ios"))))]
pub type BabyBearDft = p3_dft::Radix2DFTSmallBatch<p3_baby_bear::BabyBear>;

#[cfg(all(feature = "gpu-metal", feature = "ffi"))]
pub mod ffi;
