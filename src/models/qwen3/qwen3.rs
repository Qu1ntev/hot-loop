use std::io::{Read, Seek, Write};
use std::sync::Arc;
use candle_core::quantized::gguf_file;
use candle_core::{DType, Device, Tensor};
use candle_nn::Embedding;
use candle_nn::kv_cache::ConcatKvCache;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::mimi::candle;
// use candle_examples::token_output_stream::TokenOutputStream;
use candle_transformers::models::quantized_qwen3::ModelWeights;
use super::transformers::{AttentionWeights, Gguf, RotaryEmbedding};
use thiserror::Error;
use tokenizers::tokenizer;

#[derive(Error, Debug)]
enum Error {
    #[error(transparent)]
    Candle(#[from] candle::Error),

    #[error(transparent)]
    Tokenizers(#[from] tokenizer::Error),

    #[error("is_none")]
    None
}

pub struct Qwen3 {
    weights: Vec<AttentionWeights>, // read only веса модели
}

impl Qwen3 {
    pub fn new<M: Read + Seek>(
        model: &mut M,
        tokenizers: &[u8],
        device: &Device
    ) -> Result<Self, Error> {
        let ct = gguf_file::Content::read(model)?;

        let mut gg = Gguf::new(ct, model, device.clone());
        let md_get = |s: &str| match gg.metadata().get(s) {
            None => candle_core::bail!("cannot find {s} in metadata"),
            Some(v) => Ok(v),
        };

        let num_attention_heads = md_get("qwen3.attention.head_count")?.to_u32()? as usize;
        let num_kv_heads = md_get("qwen3.attention.head_count_kv")?.to_u32()? as usize;
        let head_dim = md_get("qwen3.attention.key_length")?.to_u32()? as usize;
        let num_layers = md_get("qwen3.block_count")?.to_u32()? as usize;
        let hidden_size = md_get("qwen3.embedding_length")?.to_u32()? as usize;
        let max_position_embeddings = md_get("qwen3.context_length")?.to_u32()? as usize;
        let rms_norm_eps = md_get("qwen3.attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
        let rope_freq_base = md_get("qwen3.rope.freq_base")?.to_f32()? as f64;

        let dtype = match gg.metadata().get("general.dtype") {
            Some(v) => match v.to_u32() {
                Ok(0) => DType::F32,
                Ok(1) => DType::F16,
                _ => DType::F16,
            },
            None => DType::F16,
        };

        let embed_tensor = gg.tensor("token_embd.weight")?;
        let embed_tokens = Embedding::new(embed_tensor.dequantize(device)?, hidden_size);

        let rotary = Arc::new(RotaryEmbedding::new(
            dtype,
            head_dim,
            max_position_embeddings,
            rope_freq_base,
            device,
        )?);

        let mut layers = Vec::with_capacity(num_layers);

        for i in 0..num_layers {
            let prefix = format!("blk.{i}");

            let weights = AttentionWeights::new(
                &mut gg,
                num_attention_heads,
                //num_key_value_heads,
                num_kv_heads,
                head_dim,
                rms_norm_eps,
                rotary.clone(),
                &prefix,
            )?;

            layers.push(weights);
        }

        Ok(Self {
            weights: layers
        })
    }

    fn new_session(
        &self,
    ) -> Session {
        let kv_cache = ConcatKvCache::new(2);

        Session::new(
            &self.weights,
            kv_cache,
        )
    }
}

struct Session<'a> {
    weights: &'a [AttentionWeights],
    kv_cache: ConcatKvCache,
}

impl<'a> Session<'a> {
    fn new(
        weights: &'a [AttentionWeights],
        kv_cache: ConcatKvCache,
    ) -> Self {
        Self {
            weights,
            kv_cache
        }
    }

    pub fn generate(&self, prompt: &str) -> Inference {
        Inference::new()
    }
}

struct Inference {
    next_token:
}

impl Inference {
    fn new() -> Self {
        Self {}
    }

    pub fn next_chunk() {
        let input = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
        let logits = model.forward(&input, tokens.len() + index)?;
        let logits = logits.squeeze(0)?;

        let logits = if args.repeat_penalty == 1. {
            logits
        } else {
            let start_at = all_tokens.len().saturating_sub(args.repeat_last_n);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                args.repeat_penalty,
                &all_tokens[start_at..],
            )?
        };

        next_token = logits_processor.sample(&logits)?;
        all_tokens.push(next_token);

        if let Some(t) = tos.next_token(next_token)? {
            print!("{t}");
            std::io::stdout().flush()?;
        }

        sampled += 1;

        if next_token == eos_token {
            None
        };
    }
}