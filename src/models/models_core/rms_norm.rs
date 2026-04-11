use candle_core::{Tensor, Result, Module};
use candle_core::quantized::QTensor;

pub(crate) struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    pub fn from_qtensor(weight: QTensor, eps: f64) -> Result<Self> {
        let weight = weight.dequantize(&weight.device())?;
        Ok(Self { weight, eps })
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        candle_nn::ops::rms_norm(x, &self.weight, self.eps as f32)
    }
}