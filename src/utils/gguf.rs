use candle_transformers::models::with_tracing::QMatMul;
use candle_transformers::{quantized_nn::RmsNorm};
use candle_core::quantized::{gguf_file::{Content, Value}, QTensor};
use candle_core::{Device, Result};
use std::io::{Read, Seek};
use std::collections::HashMap;

pub(crate) struct Gguf<'a, R: Read + Seek> {
    prefix: &'a str,
    ct: &'a Content,
    reader: R,
    device: &'a Device,
}

impl<'a, R: Read + Seek> Gguf<'a, R> {
    pub fn new(
        prefix: &'a str,
        ct: &'a Content,
        reader: R,
        device: &'a Device
    ) -> Self {
        Self { prefix, ct, reader, device }
    }

    pub fn get_with_prefix(&self, key: &str) -> Result<&Value> {
        let key = format!("{}.{}", self.prefix, key);
        match self.ct.metadata.get(&key) {
            None => candle_core::bail!("cannot find {key} in metadata"),
            Some(v) => Ok(v),
        }
    }

    pub fn qmatmul(&mut self, name: &str) -> Result<QMatMul> {
        let ws = self.tensor(name)?;
        QMatMul::from_weights(ws.into())
    }

    pub fn rms_norm(&mut self, name: &str, eps: f64) -> Result<RmsNorm> {
        let ws = self.tensor(name)?;
        RmsNorm::from_qtensor(ws, eps)
    }

    pub fn metadata(&self) -> &HashMap<String, Value> {
        &self.ct.metadata
    }

    pub fn tensor(&mut self, name: &str) -> Result<QTensor> {
        self.ct.tensor(&mut self.reader, name, self.device)
    }
}