pub(crate) mod models_core;

pub(crate) mod qwen3_model;
pub use qwen3_model::qwen3::Qwen3;

pub(crate) mod phi3_model;
// pub use phi3_model:;

use std::io::{Read, Seek};
use candle_core::{DType, Device};
use crate::Model;
use crate::Error;
use crate::models::models_core::model::FromGguf;

#[non_exhaustive]
pub struct ModelBuilder<R: Read + Seek> {
    model: R,
    tokenizer: Option<Vec<u8>>,
    dtype: DType,
}

impl<R: Read + Seek> ModelBuilder<R> {
    pub fn new(model: R) -> Self {
        Self {
            model,
            tokenizer: None,
            dtype: DType::F16,
        }
    }

    pub fn with_dtype(mut self, dtype: DType) -> Self {
        self.dtype = dtype;
        self
    }

    pub fn with_tokenizer(mut self, tokenizer: Vec<u8>) -> Self {
        self.tokenizer = Some(tokenizer);
        self
    }

    pub fn load<M: Model + FromGguf>(self, device: Device) -> Result<M, Error> {
        M::from_gguf(self.model, self.tokenizer, device, self.dtype)
    }
}