//! # High-Level Rust Crate for Running Gguf Chat-Models, Uses the Candle Backend

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

pub(crate) mod utils;
pub use candle_core::Device;

pub mod models;

pub(crate) mod backend;
pub use backend::{
    error::Error,
    session,
    settings,
};