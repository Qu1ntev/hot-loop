use candle_core::quantized::QMatMul;
use super::super::models_core::rms_norm::RmsNorm;
use candle_core::quantized::gguf_file;
use candle_core::{DType, Device, Result as CandleResult, Tensor};
use candle_nn::{Embedding, Module};
use std::io::{Read, Seek};
use crate::Error;
use crate::utils::kv_cache::KvCache;
use tokenizers::Tokenizer;
use super::ChatFormat;
use crate::session::history::Message;
use super::super::models_core::model::ModelWeights;
use super::super::models_core::rotary_embedding::RotaryEmbedding;
use super::super::models_core::mask::mask;
use crate::utils::gguf::Gguf;
use super::transformers::LayerWeights;

pub struct Qwen3 {
    embed_tokens: Embedding,
    rotary_embedding: RotaryEmbedding,
    layers: Vec<LayerWeights>,
    norm: RmsNorm,
    lm_head: QMatMul,
    device: Device,
    dtype: DType,
    chat_format: ChatFormat,
    tokenizer: Tokenizer,
}

impl Qwen3 {
    pub fn load<M, T>(
        mut model: M,
        tokenizer: T,
        device: Device,
    ) -> Result<Self, Error>
    where
        M: Read + Seek,
        T: AsRef<[u8]>,
    {
        let ct = gguf_file::Content::read(&mut model)?;
        let tokenizer = Tokenizer::from_bytes(tokenizer)?;
        let dtype = DType::F16;

        let mut gg = Gguf::new("qwen3", &ct, model, &device);

        let num_attention_heads = gg.get_with_prefix("attention.head_count")?.to_u32()? as usize;
        let num_kv_heads = gg.get_with_prefix("attention.head_count_kv")?.to_u32()? as usize;
        let head_dim = gg.get_with_prefix("attention.key_length")?.to_u32()? as usize;
        let num_layers = gg.get_with_prefix("block_count")?.to_u32()? as usize;
        let hidden_size = gg.get_with_prefix("embedding_length")?.to_u32()? as usize;
        let max_position_embeddings = gg.get_with_prefix("context_length")?.to_u32()? as usize;
        let rms_norm_eps = gg.get_with_prefix("attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
        let rope_freq_base = gg.get_with_prefix("rope.freq_base")?.to_f32()? as f64;

        let embed_tensor = gg.tensor("token_embd.weight")?;
        let embed_tokens = Embedding::new(embed_tensor.dequantize(&device)?, hidden_size);

        let rotary_embedding = RotaryEmbedding::new(
            head_dim,
            rope_freq_base,
            max_position_embeddings,
            dtype,
            &device,
        )?;

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            layers.push(LayerWeights::new(
                &mut gg,
                num_attention_heads,
                num_kv_heads,
                head_dim,
                rms_norm_eps,
                i,
            )?);
        }

        let norm = gg.rms_norm("output_norm.weight", rms_norm_eps)?;

        let lm_head_tensor = match gg.tensor("output.weight") {
            Ok(tensor) => tensor,
            Err(_) => gg.tensor("token_embd.weight")?,
        };

        let lm_head = QMatMul::from_arc(lm_head_tensor.into())?;

        let chat_format = ChatFormat::new(&tokenizer)?;

        Ok(Self {
            embed_tokens,
            rotary_embedding,
            layers,
            norm,
            lm_head,
            device,
            dtype,
            chat_format,
            tokenizer
        })
    }
}

impl ModelWeights for Qwen3 {
    fn forward(&self, input: &Tensor, offset: usize, kv_cache: &mut KvCache) -> CandleResult<Tensor> {
        let (b, l) = input.dims2()?;
        let mut h = self.embed_tokens.forward(input)?;

        let causal_mask = if l == 1 {
            None
        } else {
            Some(mask(b, l, offset, None, self.dtype, &self.device)?)
        };

        for (layer, cache) in self.layers.iter().zip(kv_cache.iter_mut()) {
            h = layer.forward(&h, causal_mask.as_ref(), offset, &self.rotary_embedding, cache)?;
        }

        let h = self.norm.forward(&h)?;
        let last_hidden = h.narrow(1, l - 1, 1)?;
        self.lm_head.forward(&last_hidden)?.squeeze(1)
    }

    fn layers_len(&self) -> usize {
        self.layers.len()
    }

    fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn fmt_history(&self, history: &[Message]) -> Result<Vec<u32>, Error> {
        self.chat_format.fmt_history(&self.tokenizer, history)
    }

    fn eos_token(&self) -> u32 {
        self.chat_format.eos_token()
    }
}