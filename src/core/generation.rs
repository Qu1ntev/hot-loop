use candle_core::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use crate::{Error, Settings, ModelWeights, KvCache};
use crate::utils::token_output_stream::TokenOutputStream;

#[non_exhaustive]
pub struct Generation<'a, 'b, M: ModelWeights> {
    pub(crate) model: &'b M,
    pub(crate) index: usize,
    pub(crate) next_token: u32,
    pub(crate) tokens: Vec<u32>,
    pub(crate) all_tokens: Vec<u32>,
    pub(crate) parameters: Settings,
    pub(crate) device: &'b Device,
    pub(crate) eos_token: u32,
    pub(crate) logits_processor: LogitsProcessor,
    pub(crate) tos: &'a mut TokenOutputStream<'b>,
    pub(crate) kv_cache: &'a mut Vec<KvCache>
}

impl<'a, 'b, M: ModelWeights> Generation<'a, 'b, M> {
    pub fn next_chunk(&mut self) -> Result<Option<String>, Error> {
        loop {
            if self.parameters.sample_len <= self.index || self.next_token == self.eos_token {
                return Ok(None);
            }

            let current_pos = self.kv_cache
                .get(0)
                .ok_or(Error::None)?
                .current_seq_len();

            let logits = if self.index == 0 {
                let input = Tensor::new(self.tokens.as_slice(), &self.device)?.unsqueeze(0)?;
                self.model.forward(&input, current_pos, &mut self.kv_cache)?
            } else {
                let input = Tensor::new(&[self.next_token], self.device)?.unsqueeze(0)?;
                self.model.forward(&input, current_pos, &mut self.kv_cache)?
            };

            // let logits = if self.index == 0 {
            //     let input = Tensor::new(self.tokens.as_slice(), &self.device)?.unsqueeze(0)?;
            //     self.model.forward(&input, 0, &mut self.kv_cache)?
            // } else {
            //     let input = Tensor::new(&[self.next_token], self.device)?.unsqueeze(0)?;
            //     self.model.forward(&input, self.tokens.len() + self.index, &mut self.kv_cache)?
            // };

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

impl<'a, 'b, M: ModelWeights> Drop for Generation<'a, 'b, M> {
    fn drop(&mut self) {
        self.tos.clear();
    }
}