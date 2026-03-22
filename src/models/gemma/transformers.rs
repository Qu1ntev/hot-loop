use candle_transformers::quantized_nn::RmsNorm;
use candle_core::quantized::QTensor;
use candle_core::{Result, Tensor};
use candle_nn::Module;
use candle_transformers::utils::repeat_kv;
use crate::KvCache;
use std::sync::Arc;
use crate::transformers::rotary_embedding::RotaryEmbedding;

pub const MAX_SEQ_LEN: usize = 131072;
// pub const DEFAULT_SLIDING_WINDOW_TYPE: usize = 6;
// pub const DEFAULT_ROPE_FREQUENCY: f32 = 1_000_000.;
// pub const DEFAULT_ROPE_FREQUENCY_SLIDING: f32 = 10_000.;
// pub const DEFAULT_ROPE_FREQUENCY_SCALE_FACTOR: f32 = 1.;

#[derive(Debug, Clone)]
pub struct QMatMul {
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
pub struct Mlp {
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

#[derive(Clone)]
pub struct LayerWeights {
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

    pub rotary_embedding: Arc<RotaryEmbedding>,
}

impl LayerWeights {
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
            .apply(&q, &k, index_pos)?;

        let (k, v) = kv_cache.append(&k, &v)?;

        // Repeat KV for GQA
        let k = repeat_kv(k, self.n_head / self.n_kv_head)?;
        let v = repeat_kv(v, self.n_head / self.n_kv_head)?;

        // Scaled Dot-Product Attention
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut attn_weights = (q.matmul(&k.transpose(2, 3)?)? * scale)?;

        if let Some(m) = mask {
            let m_dtype = m.dtype();
            let scores_dtype = attn_weights.dtype();
            let mask = if m_dtype != scores_dtype {
                m.to_dtype(scores_dtype)?
            } else {
                m.clone()
            };
            attn_weights = attn_weights.broadcast_add(&mask)?;
        }

        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v)?;

        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape((b_sz, seq_len, self.q_dim))?;

        self.attention_wo.forward(&attn_output)
    }
}