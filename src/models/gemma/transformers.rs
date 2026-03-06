use candle_transformers::{quantized_nn::RmsNorm, utils::repeat_kv};
use candle_core::quantized::QTensor;
use candle_core::D;
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Module};
use crate::{
    KvCache
};
// use candle_transformers::models;

pub(crate) const MAX_SEQ_LEN: usize = 131072; // Gemma 3 supports 128K context window
pub(crate) const DEFAULT_SLIDING_WINDOW_TYPE: usize = 6;
pub(crate) const DEFAULT_ROPE_FREQUENCY: f32 = 1_000_000.;
pub(crate) const DEFAULT_ROPE_FREQUENCY_SLIDING: f32 = 10_000.;
pub(crate) const DEFAULT_ROPE_FREQUENCY_SCALE_FACTOR: f32 = 1.;

#[derive(Debug, Clone)]
pub(crate) struct QMatMul {
    inner: candle_core::quantized::QMatMul,
}

impl QMatMul {
    pub fn from_qtensor(qtensor: QTensor) -> Result<Self> {
        let inner = candle_core::quantized::QMatMul::from_qtensor(qtensor)?;
        Ok(Self { inner })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.inner.forward(xs)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Mlp {
    pub feed_forward_gate: QMatMul, // ffn_gate in GGUF
    pub feed_forward_up: QMatMul,   // ffn_up in GGUF
    pub feed_forward_down: QMatMul, // ffn_down in GGUF
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self.feed_forward_gate.forward(xs)?;
        let up = self.feed_forward_up.forward(xs)?;
        let silu = candle_nn::ops::silu(&gate)?;
        let gated = (silu * up)?;
        self.feed_forward_down.forward(&gated)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    pub fn new(head_dim: usize, rope_frequency: f32, device: &Device) -> Result<Self> {
        let theta: Vec<_> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / rope_frequency.powf(i as f32 / head_dim as f32))
            .collect();
        let theta = Tensor::new(theta.as_slice(), device)?;
        let idx_theta = Tensor::arange(0, MAX_SEQ_LEN as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((MAX_SEQ_LEN, 1))?
            .matmul(&theta.reshape((1, theta.elem_count()))?)?;
        let cos = idx_theta.cos()?;
        let sin = idx_theta.sin()?;
        Ok(Self { sin, cos })
    }

    fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        index_pos: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let cos = self.cos.narrow(0, index_pos, seq_len)?;
        let sin = self.sin.narrow(0, index_pos, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

#[derive(Debug, Clone)]
pub(crate) struct LayerWeights {
    // Attention components
    pub attention_wq: QMatMul,
    pub attention_wk: QMatMul,
    pub attention_wv: QMatMul,
    pub attention_wo: QMatMul,

    // Specialized normalization for Q and K
    pub attention_q_norm: RmsNorm,
    pub attention_k_norm: RmsNorm,

    // Layer normalization
    pub attention_norm: RmsNorm,      // Applied before attention
    pub post_attention_norm: RmsNorm, // Applied after attention
    pub ffn_norm: RmsNorm,            // Applied before feedforward
    pub post_ffn_norm: RmsNorm,       // Applied after feedforward

    // Feed-forward network
    pub mlp: Mlp,

    // Attention parameters
    pub n_head: usize,    // Number of query heads
    pub n_kv_head: usize, // Number of key-value heads
    pub head_dim: usize,  // Dimension of each head
    pub q_dim: usize,     // Total dimension for queries

    pub sliding_window_size: Option<usize>,

    pub rotary_embedding: RotaryEmbedding,
    pub neg_inf: Tensor,

    // Cache
    // pub kv_cache: Option<(Tensor, Tensor)>,
}

impl LayerWeights {
    pub fn mask(
        &self,
        b_sz: usize,
        seq_len: usize,
        index_pos: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor> {
        let mask: Vec<_> = if let Some(sliding_window_size) = self.sliding_window_size {
            (0..seq_len)
                .flat_map(|i| {
                    (0..seq_len).map(move |j| {
                        if i < j || j + sliding_window_size < i {
                            0u32
                        } else {
                            1u32
                        }
                    })
                })
                .collect()
        } else {
            (0..seq_len)
                .flat_map(|i| (0..seq_len).map(move |j| if i < j { 0u32 } else { 1u32 }))
                .collect()
        };
        let mask = Tensor::from_slice(&mask, (seq_len, seq_len), device)?;
        let mask = if index_pos > 0 {
            let mask0 = Tensor::zeros((seq_len, index_pos), mask.dtype(), device)?;
            Tensor::cat(&[&mask0, &mask], D::Minus1)?
        } else {
            mask
        };
        mask.expand((b_sz, 1, seq_len, seq_len + index_pos))?
            .to_dtype(dtype)
    }

    pub fn forward_attn(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        index_pos: usize,
        kv_cache: &mut KvCache
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;

        let q = self.attention_wq.forward(x)?;
        let k = self.attention_wk.forward(x)?;
        let v = self.attention_wv.forward(x)?;

        let q = q
            .reshape((b_sz, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?;

        let q = self.attention_q_norm.forward(&q.contiguous()?)?;
        let k = self.attention_k_norm.forward(&k.contiguous()?)?;

        let (q, k) = self
            .rotary_embedding
            .apply_rotary_emb_qkv(&q, &k, index_pos)?;

        // let (k, v) = match kv_cache {
        //     None => (k, v),
        //     Some((k_cache, v_cache)) => {
        //         if index_pos == 0 {
        //             (k, v)
        //         } else {
        //             let k = Tensor::cat(&[k_cache, &k], 2)?; // concat on seq dim
        //             let v = Tensor::cat(&[v_cache, &v], 2)?;
        //             (k, v)
        //         }
        //     }
        // };
        // self.kv_cache = Some((k.clone(), v.clone())); // update cache

        let (k, v) = kv_cache.append(&k, &v)?;

        // Repeat KV for GQA
        let k = repeat_kv(k, self.n_head / self.n_kv_head)?;
        let v = repeat_kv(v, self.n_head / self.n_kv_head)?;

        // Scaled Dot-Product Attention
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut attn_weights = (q.matmul(&k.transpose(2, 3)?)? * scale)?;

        if let Some(mask) = mask {
            let mask = mask.broadcast_as(attn_weights.shape())?;
            let neg_inf = self.neg_inf.broadcast_as(attn_weights.dims())?;
            attn_weights = mask.eq(0u32)?.where_cond(&neg_inf, &attn_weights)?;
        }

        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v)?;

        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape((b_sz, seq_len, self.q_dim))?;

        self.attention_wo.forward(&attn_output)
    }
}