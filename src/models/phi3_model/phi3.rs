use std::io::{Read, Seek};
use candle_core::{DType, Device, Result as CandleResult, Tensor, D};
use candle_core::quantized::{gguf_file, QMatMul};
use candle_core::quantized::tokenizer::TokenizerFromGguf;
use candle_core::IndexOp;
use candle_nn::{Embedding, Module};
use tokenizers::Tokenizer;
use crate::Error;
use crate::models::models_core::model::ChatTemplate;
use crate::models::models_core::rms_norm::RmsNorm;
use crate::models::models_core::rotary_embedding::RotaryEmbedding;
use crate::utils::gguf::Gguf;
use crate::utils::kv_cache::KvCache;
use super::super::models_core::model::FromGguf;
use super::super::models_core::model::ModelWeights;
use super::ChatFormat;
use super::transformers::LayerWeights;
// use super::super::models_core::mlp::Mlp;
use super::super::models_core::mask::mask;

pub(crate) struct Mlp {
    ffn_up: QMatMul,
    ffn_down: QMatMul,
    i_size: usize,
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let up_states = xs.apply(&self.ffn_up)?;
        let gate = up_states.narrow(D::Minus1, 0, self.i_size)?;
        let up_states = up_states.narrow(D::Minus1, self.i_size, self.i_size)?;
        let up_states = (up_states * gate.silu()?)?;
        up_states.apply(&self.ffn_down)
    }
}

#[non_exhaustive]
pub struct Phi3 {
    embed_tokens: Embedding,
    rotary_embedding: RotaryEmbedding,
    layers: Vec<LayerWeights>,

    norm: RmsNorm,
    output: QMatMul,

    device: Device,
    dtype: DType,
    chat_format: ChatFormat,
    tokenizer: Tokenizer,
}

impl FromGguf for Phi3 {
    fn from_gguf<R: Read + Seek>(
        mut model: R,
        tokenizer: Option<Vec<u8>>,
        device: Device,
        dtype: DType
    ) -> Result<Self, Error> {
        let ct = gguf_file::Content::read(&mut model)?;

        let tokenizer = match tokenizer {
            Some(tokenizer) => Tokenizer::from_bytes(tokenizer)?,
            None => TokenizerFromGguf::from_gguf(&ct)?
        };
        let chat_format = ChatFormat::new(&tokenizer)?;
        let mut gg = Gguf::new("phi3", &ct, model, &device);

        let head_count = gg.get_with_prefix("attention.head_count")?.to_u32()? as usize;
        let head_count_kv = gg.get_with_prefix("attention.head_count_kv")?.to_u32()? as usize;
        let block_count = gg.get_with_prefix("block_count")?.to_u32()? as usize;
        let embedding_length = gg.get_with_prefix("embedding_length")?.to_u32()? as usize;
        let max_seq_len = gg.get_with_prefix("context_length")?.to_u32()? as usize;
        let i_size = gg.get_with_prefix("feed_forward_length")?.to_u32()? as usize;
        // let rope_dim = gg.get_with_prefix("rope.dimension_count")?.to_u32()? as usize;
        let rms_eps = gg.get_with_prefix("attention.layer_norm_rms_epsilon")?.to_f32()? as f64;

        let head_dim = embedding_length / head_count;
        let rope_freq_base = 10_000.;

        let neg_inf = Tensor::new(f32::NEG_INFINITY, &device)?;

        let tok_embeddings = gg.tensor("token_embd.weight")?;
        let embed_tokens = Embedding::new(tok_embeddings.dequantize(&device)?, embedding_length);

        let rotary_embedding = RotaryEmbedding::new(
            head_dim,
            rope_freq_base,
            max_seq_len,
            dtype,
            &device,
        )?;

        let mut layers = Vec::with_capacity(block_count);
        for layer_idx in 0..block_count {
            let prefix = format!("blk.{layer_idx}");

            let ffn_up = QMatMul::from_arc(gg.tensor(&format!("{prefix}.ffn_up.weight"))?.into())?;
            let ffn_down = QMatMul::from_arc(gg.tensor(&format!("{prefix}.ffn_down.weight"))?.into())?;

            let mlp = Mlp {
                ffn_up,
                ffn_down,
                i_size,
            };

            layers.push(LayerWeights::new(
                &mut gg,
                head_count,
                head_count_kv,
                head_dim,
                rms_eps,
                neg_inf.clone(),
                mlp,
                &prefix,
            )?);
        }

        let norm = gg.rms_norm("output_norm.weight", rms_eps)?;
        let lm_head_tensor = gg.tensor("output.weight")?;
        let output = QMatMul::from_arc(lm_head_tensor.into())?;

        Ok(Self {
            embed_tokens,
            rotary_embedding,

            layers,

            norm,
            output,

            device,
            dtype,
            chat_format,
            tokenizer,
        })
    }
}

impl ModelWeights for Phi3 {
    fn forward(&self, input: &Tensor, offset: usize, kv_cache: &mut KvCache) -> CandleResult<Tensor> {
        let (b_sz, seq_len) = input.dims2()?;

        let mask = if seq_len == 1 {
            None
        } else {
            Some(mask(b_sz, seq_len, offset, None, self.dtype, &self.device)?)
        };

        let mut input = self.embed_tokens.forward(input)?;

        for (layer, cache) in self.layers.iter().zip(kv_cache.iter_mut()) {
            let residual = &input;
            let ys = input.apply(&layer.attn_norm)?;
            let ys = layer.forward(&ys, mask.as_ref(), offset, &self.rotary_embedding, cache)?;
            let ys = (ys + residual)?;
            let residual = &ys;
            let ys = ys.apply(&layer.ffn_norm)?;
            let ys = layer.mlp.forward(&ys)?;
            input = (ys + residual)?
        }

        let input = input.apply(&self.norm)?.i((.., seq_len - 1, ..))?;
        self.output.forward(&input)
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

    fn chat_format(&self) -> &impl ChatTemplate {
        &self.chat_format
    }
}