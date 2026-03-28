use candle_core::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use crate::{
    Error, Model, settings::Settings,
};
use crate::utils::token_output_stream::TokenOutputStream;
use crate::utils::kv_cache::KvCache;

#[non_exhaustive]
pub struct Generation<'session, 'model, M: Model> {
    pub(crate) model: &'model M,
    pub(crate) index: usize,
    pub(crate) next_token: u32,
    pub(crate) tokens: Vec<u32>,
    pub(crate) all_tokens: Vec<u32>,
    pub(crate) parameters: Settings,
    pub(crate) device: &'model Device,
    pub(crate) eos_token: u32,
    pub(crate) logits_processor: LogitsProcessor,
    pub(crate) tos: &'session mut TokenOutputStream<'model>,
    pub(crate) kv_cache: &'session mut KvCache
}

impl<'session, 'model, M: Model> Generation<'session, 'model, M> {
    pub fn next_chunk(&mut self) -> Result<Option<String>, Error> {
        loop {
            if self.parameters.sample_len <= self.index || self.next_token == self.eos_token {
                return Ok(None);
            }

            let current_pos = self.kv_cache.current_pos();

            let input = if self.index == 0 {
                Tensor::new(self.tokens.as_slice(), &self.device)?.unsqueeze(0)?
            } else {
                Tensor::new(&[self.next_token], self.device)?.unsqueeze(0)?
            };

            let logits = self.model.forward(&input, current_pos, &mut self.kv_cache)?;
            let logits = logits.squeeze(0)?;

            let logits = if self.parameters.repeat_penalty == 1. {
                logits
            } else {
                let start_at = self.all_tokens.len().saturating_sub(self.parameters.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.parameters.repeat_penalty,
                    &self.all_tokens[start_at..],
                )?
            };

            self.next_token = self.logits_processor.sample(&logits)?;
            self.all_tokens.push(self.next_token);

            self.index += 1;

            if let Some(chunk) = self.tos.next_token(self.next_token)? {
                return Ok(Some(chunk))
            }
        }
    }
}

impl<'session, 'model, M: Model> Drop for Generation<'session, 'model, M> {
    fn drop(&mut self) {
        self.tos.clear();
    }
}