use candle_transformers::models::with_tracing::QMatMul;
use candle_transformers::{quantized_nn::RmsNorm};
use candle_core::quantized::{gguf_file};
use candle_core::{DType, Device, Result as CandleResult, Tensor};
use candle_nn::{Embedding, Module};
use std::io::{Read, Seek};
use std::sync::Arc;
use crate::{ModelWeights, KvCache, Error, Role, transformers::{RotaryEmbedding, mask}};
use tokenizers::Tokenizer;
use super::ChatTemplate;

use super::{
    transformers::{
        LayerWeights,
        Gguf,
    }
};

#[derive(Clone)]
pub struct Qwen3 {
    embed_tokens: Embedding,
    layers: Vec<LayerWeights>,
    norm: RmsNorm,
    lm_head: QMatMul,
    device: Device,
    dtype: DType,
    chat_template: ChatTemplate
}

impl Qwen3 {
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
            head_dim,
            rope_freq_base,
            max_position_embeddings,
            dtype,
            device,
        )?);

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            layers.push(LayerWeights::new(
                &mut gg,
                num_attention_heads,
                num_kv_heads,
                head_dim,
                rms_norm_eps,
                rotary.clone(),
                i,
            )?);
        }

        let norm = gg.rms_norm("output_norm.weight", rms_norm_eps)?;
        // Load output projection tensor, falling back to tied embeddings like gemma3
        let lm_head_tensor = match gg.tensor("output.weight") {
            Ok(tensor) => tensor,
            Err(_) => gg.tensor("token_embd.weight")?,
        };
        let lm_head = QMatMul::from_weights(lm_head_tensor.into())?;

        let chat_template = ChatTemplate::new(tokenizer)?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: device.clone(),
            dtype,
            chat_template
        })
    }
}

impl ModelWeights for Qwen3 {
    fn forward(&self, input: &Tensor, offset: usize, kv_cache: &mut Vec<KvCache>) -> CandleResult<Tensor> {
        let (b, l) = input.dims2()?;
        let mut h = self.embed_tokens.forward(input)?;
        let causal_mask = if l == 1 {
            None
        } else {
            Some(mask(b, l, offset, None, self.dtype, &self.device)?)
        };
        
        for (layer, cache) in self.layers.iter().zip(kv_cache.iter_mut()) {
            h = layer.forward(&h, causal_mask.as_ref(), offset, cache)?;
        }

        let h = self.norm.forward(&h)?;
        let last_hidden = h.narrow(1, l - 1, 1)?;
        self.lm_head.forward(&last_hidden)?.squeeze(1)
    }

    fn layers_len(&self) -> usize {
        self.layers.len()
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