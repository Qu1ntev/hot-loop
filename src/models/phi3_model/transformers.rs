use std::io::{Read, Seek};
use candle_core::{Module, Result, Tensor, D};
use super::super::models_core::rms_norm::RmsNorm;
// use super::super::models_core::mlp::Mlp;
use candle_core::quantized::QMatMul;
// use candle_nn::Activation;
use candle_transformers::utils::repeat_kv;
use crate::models::models_core::rotary_embedding::RotaryEmbedding;
use crate::utils::gguf::Gguf;
use crate::utils::kv_cache::ConcatKvCache;
use super::phi3::Mlp;

pub(crate) struct LayerWeights {
    qkv_proj: QMatMul,

    attn_output: QMatMul,

    pub attn_norm: RmsNorm,
    pub ffn_norm: RmsNorm,

    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,

    neg_inf: Tensor,
    pub mlp: Mlp
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: &Tensor) -> Result<Tensor> {
    let shape = mask.shape();
    let m = mask.where_cond(&on_true.broadcast_as(shape.dims())?, on_false)?;
    Ok(m)
}

impl LayerWeights {
    pub fn new<R: Read + Seek>(
        gg: &mut Gguf<R>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rms_norm_eps: f64,

        neg_inf: Tensor,
        mlp: Mlp,
        prefix: &str,
    ) -> Result<Self> {
        let num_kv_groups = num_heads / num_kv_heads;

        let qkv_proj = gg.qmatmul(&format!("{prefix}.attn_qkv.weight"))?;
        let attn_output = gg.qmatmul(&format!("{prefix}.attn_output.weight"))?;

        let attn_norm = gg.rms_norm(&format!("{prefix}.attn_norm.weight"), rms_norm_eps)?;
        let ffn_norm = gg.rms_norm(&format!("{prefix}.ffn_norm.weight"), rms_norm_eps)?;

        Ok(Self {
            qkv_proj,
            attn_output,
            attn_norm,
            ffn_norm,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,

            neg_inf,
            mlp
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        attn_mask: Option<&Tensor>,
        offset: usize,
        rotary_embedding: &RotaryEmbedding,
        kv_cache: &mut ConcatKvCache
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;

        let qkv = self.qkv_proj.forward(x)?;

        let query_pos = self.num_heads * self.head_dim;
        let q = qkv.narrow(D::Minus1, 0, query_pos)?;
        let k = qkv.narrow(D::Minus1, query_pos, self.num_kv_heads * self.head_dim)?;
        let v = qkv.narrow(
            D::Minus1,
            query_pos + self.num_kv_heads * self.head_dim,
            self.num_kv_heads * self.head_dim,
        )?;

        let q = q
            .reshape((b_sz, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (q, k) = rotary_embedding.apply(&q, &k, offset)?;

        let (k, v) = kv_cache.append(&k, &v)?;

        let k = repeat_kv(k, self.num_kv_groups)?.contiguous()?; // .contiguous() test
        let v = repeat_kv(v, self.num_kv_groups)?.contiguous()?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        if let Some(m) = attn_mask {
            let m_dtype = m.dtype();
            let scores_dtype = scores.dtype();
            let mask = if m_dtype != scores_dtype {
                m.to_dtype(scores_dtype)?
            } else {
                m.clone()
            };
            scores = scores.broadcast_add(&mask)?;
        }

        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v)?;

        let reshaped_ctx = ctx
            .transpose(1, 2)?
            .reshape((b_sz, seq_len, ()))?;

        self.attn_output.forward(&reshaped_ctx)
    }
}