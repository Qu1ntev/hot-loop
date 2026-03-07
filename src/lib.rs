//! # High-Level Pure-Rust Crate for Running Gguf Chat-Models, Uses the Candle Backend
//!
//! ---
//!
//! ## Easy to use:
//! ```rust
//! use std::fs::{File, read};
//! use std::io::{stdout, Write};
//!
//! use hot_loop::{
//!     Model,
//!     models::Qwen3,
//!     Device,
//!     Error,
//! };
//!
//! fn main() -> Result<(), Error> {
//!     let mut model_file = File::open("models/Qwen3-4B-it-Q4_K_M.gguf").unwrap();
//!     let tokenizer_bytes = read("models/tokenizer.json").unwrap();
//!
//!     // model read only
//!     let model = Qwen3::load(&mut model_file, &tokenizer_bytes, &Device::Cpu)?;
//!
//!     let mut session = model.new_session();
//!     // and more sessions!
//!     // let mut session2 = model.new_session();
//!     // let mut session3 = model.new_session();
//!
//!     let mut generate = session.generate("Hello!")?;
//!
//!     while let Some(chunk) = generate.next_chunk()? {
//!         print!("{chunk}");
//!         stdout().flush().unwrap();
//!     }
//!
//!     Ok(())
//! }
//! ```

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