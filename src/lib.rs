#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

pub(crate) mod utils;
pub use candle_core::Device;

pub mod models;

pub(crate) mod core;
pub use core::{
    model_weights::Model,
    error::Error,
    session,
    settings,
};
pub(crate) use core::{
    model_weights::{
        ModelWeights, KvCache, Role
    },
};