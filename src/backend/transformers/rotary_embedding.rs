use candle_core::{DType, Device, Tensor, Result};

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