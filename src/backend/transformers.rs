use candle_core::{DType, Device, Tensor, Result};
// use candle_transformers::models::with_tracing::QMatMul;
// use candle_nn::{Activation, Module};
// use std::io::{Read, Seek};
// use candle_core::quantized::{gguf_file, QTensor};
// use candle_transformers::quantized_nn::RmsNorm;
// use std::collections::HashMap;

#[derive(Clone)]
pub(crate) struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    pub fn new(
        head_dim: usize,
        rope_freq: f64,
        max_pos_embed: usize,
        dtype: DType,
        device: &Device
    ) -> Result<Self> {
        let inv_freq: Vec<_> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / rope_freq.powf(i as f64 / head_dim as f64) as f32)
            .collect();

        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), device)?.to_dtype(dtype)?;

        let t = Tensor::arange(0u32, max_pos_embed as u32, device)?
            .to_dtype(dtype)?
            .reshape((max_pos_embed, 1))?
            .matmul(&inv_freq)?;

        let sin = t.sin()?;
        let cos = t.cos()?;

        Ok(Self { sin, cos })
    }

    pub fn apply(
        &self,
        q: &Tensor,
        k: &Tensor,
        index_pos: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (_, _, seq_len, _) = q.dims4()?;
        let cos = self.cos.narrow(0, index_pos, seq_len)?.to_dtype(q.dtype())?;
        let sin = self.sin.narrow(0, index_pos, seq_len)?.to_dtype(q.dtype())?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

pub(crate) fn mask(
    b_sz: usize,
    seq_len: usize,
    offset: usize,
    sliding_window: Option<usize>,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let mask: Vec<_> = (0..seq_len)
        .flat_map(|i| {
            (0..(seq_len + offset)).map(move |j| {
                let past_ok = j <= i + offset;

                let sw_ok = match sliding_window {
                    Some(w) => (i + offset) as i64 - j as i64 <= w as i64,
                    None => true,
                };

                if past_ok && sw_ok { 0. } else { f32::NEG_INFINITY }
            })
        })
        .collect();
    Tensor::from_slice(&mask, (b_sz, 1, seq_len, seq_len + offset), device)?.to_dtype(dtype)
}

// pub(crate) struct Gguf<'a, R: Read + Seek> {
//     ct: &'a gguf_file::Content,
//     reader: &'a mut R,
//     device: &'a Device,
// }
// 
// impl<'a, R: Read + Seek> Gguf<'a, R> {
//     pub fn new(ct: &'a gguf_file::Content, reader: &'a mut R, device: &'a Device) -> Self {
//         Self { ct, reader, device }
//     }
// 
//     pub fn qmatmul(&mut self, name: &str) -> Result<QMatMul> {
//         let ws = self.ct.tensor(&mut self.reader, name, &self.device)?;
//         QMatMul::from_weights(ws.into())
//     }
// 
//     pub fn rms_norm(&mut self, name: &str, eps: f64) -> Result<RmsNorm> {
//         let ws = self.ct.tensor(&mut self.reader, name, &self.device)?;
//         RmsNorm::from_qtensor(ws, eps)
//     }
// 
//     pub fn metadata(&self) -> &HashMap<String, gguf_file::Value> {
//         &self.ct.metadata
//     }
// 
//     pub fn tensor(&mut self, name: &str) -> Result<QTensor> {
//         self.ct.tensor(&mut self.reader, name, &self.device)
//     }
// }
// 
// #[derive(Debug, Clone)]
// struct Mlp {
//     gate: QMatMul,
//     up: QMatMul,
//     down: QMatMul,
//     silu: Activation,
// }
// 
// impl Mlp {
//     fn new<R: Read + Seek>(gg: &mut Gguf<R>, prefix: &str) -> Result<Self> {
//         let gate = gg.qmatmul(&format!("{prefix}.ffn_gate.weight"))?;
//         let up = gg.qmatmul(&format!("{prefix}.ffn_up.weight"))?;
//         let down = gg.qmatmul(&format!("{prefix}.ffn_down.weight"))?;
//         let silu = Activation::Silu;
// 
//         Ok(Self {
//             gate,
//             up,
//             down,
//             silu,
//         })
//     }
// }
// 
// impl Module for Mlp {
//     fn forward(&self, x: &Tensor) -> Result<Tensor> {
//         let gate = self.gate.forward(x)?.apply(&self.silu)?;
//         let up = self.up.forward(x)?;
//         let gated = (gate * up)?;
//         self.down.forward(&gated)
//     }
// }