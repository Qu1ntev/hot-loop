use candle_transformers::models::with_tracing::QMatMul;
use candle_transformers::{quantized_nn::RmsNorm};
use candle_core::quantized::{gguf_file};
use candle_core::{DType, Device, Result as CandleResult, Tensor};
use candle_nn::{Embedding, Module};
use std::io::{Read, Seek};
use std::sync::Arc;
use tokenizers::Tokenizer;
use super::ChatTemplate;
use crate::{ModelWeights, KvCache, Error, Role};
use crate::transformers::rotary_embedding::RotaryEmbedding;
use crate::transformers::mask::mask;
use super::transformers::LayerWeights;
use crate::utils::helpers::Gguf;

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
        let mut gguf = Gguf::new("qwen3", &ct, model, device);

        let num_attention_heads = gguf.get_with_prefix("attention.head_count")?.to_u32()? as usize;
        let num_kv_heads = gguf.get_with_prefix("attention.head_count_kv")?.to_u32()? as usize;
        let head_dim = gguf.get_with_prefix("attention.key_length")?.to_u32()? as usize;
        let num_layers = gguf.get_with_prefix("block_count")?.to_u32()? as usize;
        let hidden_size = gguf.get_with_prefix("embedding_length")?.to_u32()? as usize;
        let max_position_embeddings = gguf.get_with_prefix("context_length")?.to_u32()? as usize;
        let rms_norm_eps = gguf.get_with_prefix("attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
        let rope_freq_base = gguf.get_with_prefix("rope.freq_base")?.to_f32()? as f64;

        let dtype = match ct.metadata.get("general.dtype") {
            Some(v) => match v.to_u32() {
                Ok(0) => DType::F32,
                Ok(1) => DType::F16,
                _ => DType::F16,
            },
            None => DType::F16,
        };

        let embed_tensor = gguf.tensor("token_embd.weight")?;
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
                &mut gguf,
                num_attention_heads,
                num_kv_heads,
                head_dim,
                rms_norm_eps,
                rotary.clone(),
                i,
            )?);
        }

        let norm = gguf.rms_norm("output_norm.weight", rms_norm_eps)?;

        let lm_head_tensor = match gguf.tensor("output.weight") {
            Ok(tensor) => tensor,
            Err(_) => gguf.tensor("token_embd.weight")?,
        };
        let lm_head = QMatMul::from_weights(lm_head_tensor.into())?;

        let tokenizer = Tokenizer::from_bytes(tokenizer)?;
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