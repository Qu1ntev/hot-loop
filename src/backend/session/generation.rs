// use candle_core::Tensor;
// use candle_transformers::generation::LogitsProcessor;
// use crate::settings::Settings;
// use crate::Error;
// use crate::Model;
// use crate::utils::token_output_stream::TokenOutputStream;
// use crate::utils::kv_cache::KvCache;
use std::sync::mpsc::Receiver;

#[non_exhaustive]
pub struct Generation<'session> {
    _handle: &'session (),
    receiver: Receiver<String>
}

impl<'session> Generation<'session> {
    pub(crate) fn new(_handle: &'session (), receiver: Receiver<String>) -> Self {
        Self { _handle, receiver }
    }
    
    pub fn next_chunk(&self) -> Option<String> {
        self.receiver.recv().ok()
    }
}

// #[non_exhaustive]
// pub struct Generation<'session, M: Model> {
//     model: &'session M,
//     index: usize,
//     next_token: u32,
//     tokens_prefill: Option<Vec<u32>>,
//     all_tokens: Vec<u32>,
//     settings: Settings,
//     eos_token: u32,
//     logits_processor: LogitsProcessor,
//     tos: &'session mut TokenOutputStream,
//     kv_cache: &'session mut KvCache
// }
// 
// impl<'session, M: Model> Generation<'session, M> {
//     pub(crate) fn new(
//         model: &'session M,
//         tokens_prefill: Vec<u32>,
//         logits_processor: LogitsProcessor,
//         settings: Settings,
//         tos: &'session mut TokenOutputStream,
//         kv_cache: &'session mut KvCache,
//         eos_token: u32,
//     ) -> Self {
//         Self {
//             model,
//             index: 0,
//             next_token: 0,
//             all_tokens: Vec::new(),
//             tokens_prefill: Some(tokens_prefill),
//             logits_processor,
//             settings,
//             tos,
//             kv_cache,
//             eos_token,
//         }
//     }
// 
//     pub fn next_chunk(&mut self) -> Result<Option<String>, Error> {
//         loop {
//             if self.settings.sample_len <= self.index || self.next_token == self.eos_token {
//                 return Ok(None);
//             }
// 
//             let current_pos = self.kv_cache.current_pos();
// 
//             let input = if self.index == 0 &&
//                 let Some(tokens_prefill) = self.tokens_prefill.take() {
//                 Tensor::new(tokens_prefill.as_slice(), self.model.device())?.unsqueeze(0)?
// 
//             } else {
//                 Tensor::new(&[self.next_token], self.model.device())?.unsqueeze(0)?
//             };
// 
//             let logits = self.model.forward(&input, current_pos, &mut self.kv_cache)?;
//             let logits = logits.squeeze(0)?;
// 
//             let logits = if self.settings.repeat_penalty == 1. {
//                 logits
//             } else {
//                 let start_at = self.all_tokens.len().saturating_sub(self.settings.repeat_last_n);
//                 candle_transformers::utils::apply_repeat_penalty(
//                     &logits,
//                     self.settings.repeat_penalty,
//                     &self.all_tokens[start_at..],
//                 )?
//             };
// 
//             self.next_token = self.logits_processor.sample(&logits)?;
//             self.all_tokens.push(self.next_token);
// 
//             self.index += 1;
// 
//             if let Some(chunk) = self.tos.next_token(self.model.tokenizer(), self.next_token)? {
//                 return Ok(Some(chunk))
//             }
//         }
//     }
// }
// 
// impl<'session, M: Model> Drop for Generation<'session, M> {
//     fn drop(&mut self) {
//         self.tos.clear();
//     }
// }