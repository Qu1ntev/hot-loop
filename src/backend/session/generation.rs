use candle_core::Device;
use candle_core::Tensor;
use candle_core::Result as CandleResult;
use candle_core::Error as CandleError;
use candle_transformers::generation::LogitsProcessor;
use super::history::Role;
use crate::Error;
use crate::Model;
use crate::settings::Settings;
use crate::utils::token_output_stream::TokenOutputStream;
use crate::utils::kv_cache::KvCache;

#[non_exhaustive]
pub struct Generation<'a, 'model, M: Model> {
    pub(crate) model: &'model M,
    pub(crate) index: usize,
    pub(crate) next_token: u32,
    pub(crate) tokens: Vec<u32>,
    pub(crate) all_tokens: Vec<u32>,
    pub(crate) parameters: Settings,
    pub(crate) device: &'model Device,
    pub(crate) eos_tokens: Vec<u32>,
    pub(crate) logits_processor: LogitsProcessor,
    pub(crate) tos: &'a mut TokenOutputStream<'model>,
    pub(crate) kv_cache: &'a mut Vec<KvCache>,
    pub(crate) cached_tokens: &'a mut Vec<u32>,
}

impl<'a, 'model, M: Model> Generation<'a, 'model, M> {
    // pub(crate) fn new() -> Self {
    //     
    //     
    //     Self {
    //         
    //     }
    // }
    
    pub fn next_chunk(&mut self) -> Result<Option<String>, Error> {
        loop {
            let logits = self.apply_logits()?;
            let logits = self.apply_repeat_penalty(logits)?;

            self.next_token = self.logits_processor.sample(&logits)?;
            self.all_tokens.push(self.next_token);

            self.index += 1;

            if self.is_end() {
                self.stop_with_eos()?;
                self.debug()?;
                self.commit()?;
                return Ok(None);
            }

            if self.parameters.sample_len <= self.index {
                self.stop_with_sample_len()?;
                self.debug()?;
                self.commit()?;
                return Ok(None);
            }

            if let Some(chunk) = self.tos.next_token(self.next_token)? {
                return Ok(Some(chunk))
            }
        }
    }

    fn stop_with_eos(&mut self) -> Result<(), Error> {
        let current_pos = self.current_pos()?;

        let input = Tensor::new(&[self.next_token], self.device)?.unsqueeze(0)?;
        self.model.forward(&input, current_pos, &mut self.kv_cache)?;
        Ok(())
    }

    fn stop_with_sample_len(&mut self) -> Result<(), Error> {
        let current_pos = self.current_pos()?;

        let input = Tensor::new(self.eos_tokens.as_slice(), self.device)?.unsqueeze(0)?;
        self.model.forward(&input, current_pos, &mut self.kv_cache)?;
        Ok(())
    }

    fn is_end(&self) -> bool {
        if self.eos_tokens.is_empty() || self.all_tokens.len() < self.eos_tokens.len() {
            return false;
        }
        self.all_tokens[self.all_tokens.len() - self.eos_tokens.len()..] == self.eos_tokens
    }

    fn apply_logits(&mut self) -> CandleResult<Tensor> {
        let current_pos = self.current_pos()?;

        let input = if self.index == 0 {
            Tensor::new(self.tokens.as_slice(), self.device)?.unsqueeze(0)?
        } else {
            Tensor::new(&[self.next_token], self.device)?.unsqueeze(0)?
        };

        let logits = self.model.forward(&input, current_pos, &mut self.kv_cache)?;

        logits.squeeze(0)
    }

    fn current_pos(&self) -> CandleResult<usize> {
        let current_pos = self.kv_cache
            .get(0)
            .ok_or_else(|| CandleError::Msg("kv cache missing index".into()))?
            .current_seq_len();

        Ok(current_pos)
    }

    fn apply_repeat_penalty(&mut self, logits: Tensor) -> CandleResult<Tensor> {
        if self.parameters.repeat_penalty == 1. {
            Ok(logits)
        } else {
            let start_at = self.all_tokens.len().saturating_sub(self.parameters.repeat_last_n);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                self.parameters.repeat_penalty,
                &self.all_tokens[start_at..],
            )
        }
    }
    
    fn commit(&mut self) -> Result<(), Error> {
        let text = self.model.tokenizer().decode(&self.all_tokens, false)?;
        let tokens = self.model.fmt_prompt(Role::Assistant, &text)?;
        self.cached_tokens.extend_from_slice(&tokens);
        Ok(())
    }

    fn debug(&self) -> Result<(), Error> {
        let current_pos = self.current_pos()?;
        println!("\nkv cache current pos: {}", current_pos);
        let decoded = self.model.tokenizer().decode(&self.all_tokens, false)?;
        println!("all_tokens decoded: {:?}", decoded);
        Ok(())
    }
}

impl<'a, 'model, M: Model> Drop for Generation<'a, 'model, M> {
    fn drop(&mut self) {
        self.tos.clear();
    }
}