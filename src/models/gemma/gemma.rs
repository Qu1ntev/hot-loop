use candle_transformers::{quantized_nn::RmsNorm};
use candle_core::quantized::gguf_file;
use std::io::{Read, Seek};
use candle_core::{Device, IndexOp, Result as CandleResult, Tensor};
use candle_nn::{Embedding, Module};
use tokenizers::Tokenizer;
use super::{
    transformers::{
        LayerWeights,
        QMatMul,
        DEFAULT_SLIDING_WINDOW_TYPE,
        DEFAULT_ROPE_FREQUENCY,
        DEFAULT_ROPE_FREQUENCY_SLIDING,
        DEFAULT_ROPE_FREQUENCY_SCALE_FACTOR,
        RotaryEmbedding,
        Mlp,
    },
    ChatTemplate
};
use crate::{
    Error,
    ModelWeights,
    KvCache,
    Role,
    session::Session
};

#[derive(Clone)]
pub struct Gemma {
    tok_embeddings: Embedding,
    embedding_length: usize,
    layers: Vec<LayerWeights>,
    norm: RmsNorm,
    output: QMatMul,
    device: Device,
    chat_template: ChatTemplate,
}

impl Gemma {
    pub fn load<M, T>(
        model: &mut M,
        tokenizer: T,
        device: &Device,
    ) -> Result<Self, Error>
    where
        M: Read + Seek,
        T: AsRef<[u8]>,
    {
        let ct = gguf_file::Content::read(model)?;

        let tokenizer = Tokenizer::from_bytes(tokenizer)?;

        // Detect architecture prefix by probing which keys exist in metadata.
        // This supports gemma3, gemma2, gemma, gemma-embedding, and future variants.
        let prefix = ["gemma3", "gemma2", "gemma", "gemma-embedding"]
            .iter()
            .find(|p| {
                ct.metadata
                    .contains_key(&format!("{}.attention.head_count", p))
            })
            .copied()
            .unwrap_or("gemma3");

        let md_get = |s: &str| {
            let key = format!("{prefix}.{s}");
            match ct.metadata.get(&key) {
                None => candle_core::bail!("cannot find {key} in metadata"),
                Some(v) => Ok(v),
            }
        };

        let head_count = md_get("attention.head_count")?.to_u32()? as usize;
        let head_count_kv = md_get("attention.head_count_kv")?.to_u32()? as usize;
        let block_count = md_get("block_count")?.to_u32()? as usize;
        let embedding_length = md_get("embedding_length")?.to_u32()? as usize;
        let key_length = md_get("attention.key_length")?.to_u32()? as usize;
        let _value_length = md_get("attention.value_length")?.to_u32()? as usize;
        let rms_norm_eps = md_get("attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
        let sliding_window_size = md_get("attention.sliding_window")?.to_u32()? as usize;

        let sliding_window_type = md_get("attention.sliding_window_type")
            .and_then(|m| Ok(m.to_u32()? as usize))
            .unwrap_or(DEFAULT_SLIDING_WINDOW_TYPE);

        let rope_freq_base = md_get("rope.freq_base")
            .and_then(|m| m.to_f32())
            .unwrap_or(DEFAULT_ROPE_FREQUENCY);

        let rope_freq_base_sliding = md_get("rope.local_freq_base")
            .and_then(|m| m.to_f32())
            .unwrap_or(DEFAULT_ROPE_FREQUENCY_SLIDING);

        // Unused in Llama.cpp so we aren't using it here.
        let _rope_freq_scaling_factor = md_get("rope.scaling.factor")
            .and_then(|m| m.to_f32())
            .unwrap_or(DEFAULT_ROPE_FREQUENCY_SCALE_FACTOR);

        // Compute the dimensions for queries, keys, and values
        // These are the total dimensions when projected across all heads
        let q_dim = head_count * key_length;

        let neg_inf = Tensor::new(f32::NEG_INFINITY, device)?;

        // Load token embeddings and output projection
        let tok_embeddings = ct.tensor(model, "token_embd.weight", device)?;
        let tok_embeddings = tok_embeddings.dequantize(device)?;
        let norm = RmsNorm::from_qtensor(
            ct.tensor(model, "output_norm.weight", device)?,
            rms_norm_eps,
        )?;
        let output = match ct.tensor(model, "output.weight", device) {
            Ok(tensor) => tensor,
            Err(_) => ct.tensor(model, "token_embd.weight", device)?, // Use tied weights if output.weight doesn't exist
        };

        let mut layers = Vec::with_capacity(block_count);
        for layer_idx in 0..block_count {
            let prefix = format!("blk.{layer_idx}");

            let attention_wq = ct.tensor(model, &format!("{prefix}.attn_q.weight"), device)?;
            let attention_wk = ct.tensor(model, &format!("{prefix}.attn_k.weight"), device)?;
            let attention_wv = ct.tensor(model, &format!("{prefix}.attn_v.weight"), device)?;
            let attention_wo =
                ct.tensor(model, &format!("{prefix}.attn_output.weight"), device)?;

            let attention_q_norm = RmsNorm::from_qtensor(
                ct.tensor(model, &format!("{prefix}.attn_q_norm.weight"), device)?,
                rms_norm_eps,
            )?;

            let attention_k_norm = RmsNorm::from_qtensor(
                ct.tensor(model, &format!("{prefix}.attn_k_norm.weight"), device)?,
                rms_norm_eps,
            )?;

            let attention_norm = RmsNorm::from_qtensor(
                ct.tensor(model, &format!("{prefix}.attn_norm.weight"), device)?,
                rms_norm_eps,
            )?;

            let post_attention_norm = RmsNorm::from_qtensor(
                ct.tensor(
                    model,
                    &format!("{prefix}.post_attention_norm.weight"),
                    device,
                )?,
                rms_norm_eps,
            )?;

            let ffn_norm = RmsNorm::from_qtensor(
                ct.tensor(model, &format!("{prefix}.ffn_norm.weight"), device)?,
                rms_norm_eps,
            )?;

            let post_ffn_norm = RmsNorm::from_qtensor(
                ct.tensor(model, &format!("{prefix}.post_ffw_norm.weight"), device)?,
                rms_norm_eps,
            )?;

            let feed_forward_gate =
                ct.tensor(model, &format!("{prefix}.ffn_gate.weight"), device)?;
            let feed_forward_up = ct.tensor(model, &format!("{prefix}.ffn_up.weight"), device)?;
            let feed_forward_down =
                ct.tensor(model, &format!("{prefix}.ffn_down.weight"), device)?;

            let mlp = Mlp {
                feed_forward_gate: QMatMul::from_qtensor(feed_forward_gate)?,
                feed_forward_up: QMatMul::from_qtensor(feed_forward_up)?,
                feed_forward_down: QMatMul::from_qtensor(feed_forward_down)?,
            };

            // Sliding window pattern hardcoded to 6 because it's not explicitly defined
            let is_sliding = (layer_idx + 1) % sliding_window_type > 0;
            let sliding_window_size = is_sliding.then_some(sliding_window_size);
            let layer_rope_frequency = if is_sliding {
                rope_freq_base_sliding
            } else {
                rope_freq_base
            };

            let rotary_embedding = RotaryEmbedding::new(key_length, layer_rope_frequency, device)?;

            layers.push(LayerWeights {
                attention_wq: QMatMul::from_qtensor(attention_wq)?,
                attention_wk: QMatMul::from_qtensor(attention_wk)?,
                attention_wv: QMatMul::from_qtensor(attention_wv)?,
                attention_wo: QMatMul::from_qtensor(attention_wo)?,
                attention_q_norm,
                attention_k_norm,
                attention_norm,
                post_attention_norm,
                ffn_norm,
                post_ffn_norm,
                mlp,
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim: key_length,
                q_dim,
                sliding_window_size,
                rotary_embedding,
                neg_inf: neg_inf.clone(),
            })
        }

        let chat_template = ChatTemplate::new(tokenizer)?;

        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, embedding_length),
            embedding_length,
            layers,
            norm,
            output: QMatMul::from_qtensor(output)?,
            device: device.clone(),
            chat_template
        })
    }

    pub fn new_session(&self) -> Session<'_, Gemma> {
        Session::new(self)
    }
}

impl ModelWeights for Gemma {
    fn forward(&self, x: &Tensor, index_pos: usize, kv_cache: &mut Vec<KvCache>) -> CandleResult<Tensor> {
        let (b_sz, seq_len) = x.dims2()?;

        let mut layer_in = self.tok_embeddings.forward(x)?;
        layer_in = (layer_in * (self.embedding_length as f64).sqrt())?;

        for (layer, cache) in self.layers.iter().zip(kv_cache.iter_mut()) {
            let attention_mask = if seq_len == 1 {
                None
            } else {
                Some(layer.mask(b_sz, seq_len, index_pos, x.dtype(), x.device())?)
            };

            // Attention block
            let residual = &layer_in;
            let x = layer.attention_norm.forward(&layer_in)?;
            let x = layer.forward_attn(&x, attention_mask.as_ref(), index_pos, cache)?;
            let x = layer.post_attention_norm.forward(&x)?;
            let x = (x + residual)?;

            // Feed-forward block
            let residual = &x;
            let x = layer.ffn_norm.forward(&x)?;
            let x = layer.mlp.forward(&x)?;
            let x = layer.post_ffn_norm.forward(&x)?;
            let x = (x + residual)?;

            layer_in = x;
        }

        let x = layer_in.i((.., seq_len - 1, ..))?;
        let x = self.norm.forward(&x)?;
        let output = self.output.forward(&x)?;

        Ok(output)
    }

    fn create_kv_cache(&self) -> Vec<KvCache> {
        let mut kv_cache = Vec::with_capacity(self.layers.len());

        for _ in 0..self.layers.len() {
            kv_cache.push(KvCache::new(2));
        }

        kv_cache
    }

    fn tokenizer(&self) -> &Tokenizer {
        &self.chat_template.tokenizer()
    }

    fn current_device(&self) -> &Device {
        &self.device
    }

    fn fmt_prompt(&self, prompt: &str, role: Role) -> Result<Vec<u32>, Error> {
        self.chat_template.fmt_prompt(prompt, role)
    }

    fn assistant_start_template(&self) -> Vec<u32> {
        self.chat_template.assistant_start_template()
    }

    fn eos_token(&self) -> u32 {
        self.chat_template.eos_token()
    }
}