// use std::io::{Read, Seek};
// use candle_core::Device;
// // use candle_core::quantized::gguf_file;
// use candle_core::{Device, Tensor};
// use candle_nn::kv_cache::ConcatKvCache;
// // use candle_transformers::generation::{LogitsProcessor, Sampling};
// use candle_transformers::models::quantized_qwen3::ModelWeights;
// use tokenizers::tokenizer::Tokenizer;
// use crate::utils::token_output_stream::TokenOutputStream;
// use crate::{Settings, Error, Generation, Session, ModelWeights};

// use candle_transformers::models::quantized_qwen3::ModelWeights;
// use tokenizers::Tokenizer;
// use super::qwen3_weights::Qwen3Weights;

//
// impl ModelWeights for Qwen3 {
//     fn forwards(&self, input: &Tensor, offset: usize, kv_cache: &mut ConcatKvCache) -> Tensor {
//         Tensor::new(&[54, 545], &self.device).unwrap()
//     }
// }
//
// pub struct Qwen3 {
//     model: Qwen3Weights,
//     tos: TokenOutputStream,
//     // settings: Settings,
//     device: Device
// }
//
// impl Qwen3 {
//     pub fn load<M, T>(
//         model: &mut M,
//         tokenizer: T,
//         device: Device
//     ) -> Result<Self, Error>
//     where
//         M: Read + Seek,
//         T: AsRef<[u8]>
//     {
//         let model = {
//             // let ct = gguf_file::Content::read(model)?;
//             Qwen3Weights::from_gguf(model, &device)?
//         };
//
//         let tos = {
//             let tk = Tokenizer::from_bytes(tokenizer)?;
//             TokenOutputStream::new(tk)
//         };
//
//         // let settings = Settings::default();
//
//         Ok(Self {
//             model,
//             tos,
//             // settings,
//             device
//         })
//     }
//
//     fn new_session(&self) -> Session<'_, Qwen3> {
//         Session::new(self)
//     }
// }

