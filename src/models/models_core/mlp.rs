use candle_core::{Tensor, Result, Module};
use candle_core::quantized::QMatMul;
use candle_nn::Activation;
use std::io::{Read, Seek};
use crate::utils::gguf::Gguf;

pub(crate) struct Mlp {
    gate: QMatMul,
    up: QMatMul,
    down: QMatMul,
    act_fn: Activation,
}

impl Mlp {
    pub fn new<R: Read + Seek>(
        gguf: &mut Gguf<R>,
        prefix: &str,
        act_fn: Activation
    ) -> Result<Self> {
        let gate = gguf.qmatmul(&format!("{prefix}.ffn_gate.weight"))?;
        let up = gguf.qmatmul(&format!("{prefix}.ffn_up.weight"))?;
        let down = gguf.qmatmul(&format!("{prefix}.ffn_down.weight"))?;

        Ok(Self {
            gate,
            up,
            down,
            act_fn,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate.forward(x)?.apply(&self.act_fn)?;
        let up = self.up.forward(x)?;
        let gated = (gate * up)?;
        self.down.forward(&gated)
    }
}