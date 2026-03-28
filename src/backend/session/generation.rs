use candle_core::Tensor;
use candle_core::Result as CandleResult;
use candle_core::Error as CandleError;
use candle_transformers::generation::LogitsProcessor;
use super::session::Session;
use crate::Error;
use crate::Model;
use crate::ChatFormat;
use crate::settings::Settings;

#[non_exhaustive]
pub struct Generation<'session, 'model, M: Model> {
    session: &'session mut Session<'model, M>,
    index: usize,
    next_token: u32,
    tokens_prefill: Option<Vec<u32>>,
    all_tokens: Vec<u32>,
    logits_processor: LogitsProcessor,
    settings: Settings
}

impl<'session, 'model, M: Model> Generation<'session, 'model, M> {
    pub(crate) fn new(
        session: &'session mut Session<'model, M>,
        tokens_prefill: Vec<u32>,
        logits_processor: LogitsProcessor,
        settings: Settings
    ) -> Self {
        Self {
            session,
            index: 0,
            next_token: 0,
            tokens_prefill: Some(tokens_prefill),
            all_tokens: Vec::new(),
            logits_processor,
            settings
        }
    }
    
    pub fn next_chunk(&mut self) -> Result<Option<String>, Error> {
        loop {
            let logits = self.apply_logits()?;
            let logits = self.apply_repeat_penalty(logits)?;

            self.next_token = self.logits_processor.sample(&logits)?;
            self.all_tokens.push(self.next_token);

            self.index += 1;

            if self.settings.sample_len <= self.index || self.is_end() {
                self.session.cached_tokens.extend_from_slice(self.all_tokens.as_slice());
                return Ok(None);
            }

            if let Some(chunk) = self.session.tos.next_token(self.next_token)? {
                return Ok(Some(chunk))
            }
        }
    }

    fn is_end(&self) -> bool {
        let eos_tokens = self.session.model.chat_format().eos_tokens();
        
        if eos_tokens.is_empty() || self.all_tokens.len() < eos_tokens.len() {
            return false;
        }
        self.all_tokens[self.all_tokens.len() - eos_tokens.len()..] == *eos_tokens
    }

    fn apply_logits(&mut self) -> CandleResult<Tensor> {
        let current_pos = self.current_pos()?;
        let device = self.session.model.device();

        // Prefill
        let input = if self.index == 0 &&
            let Some(tokens_prefill) = self.tokens_prefill.as_ref() {
            let logits = Tensor::new(tokens_prefill.as_slice(), device)?.unsqueeze(0)?;
            self.tokens_prefill = None;
            logits

        // Decode
        } else {
            Tensor::new(&[self.next_token], device)?.unsqueeze(0)?
        };

        let logits = self.session.model.forward(
            &input, current_pos, &mut self.session.kv_cache
        )?;

        logits.squeeze(0)
    }

    fn current_pos(&self) -> CandleResult<usize> {
        self.session.kv_cache.current_pos()
            .ok_or_else(|| CandleError::Msg("kv_cache_pos is none".into()))
    }

    fn apply_repeat_penalty(&mut self, logits: Tensor) -> CandleResult<Tensor> {
        if self.settings.repeat_penalty == 1. {
            Ok(logits)
        } else {
            let start_at = self.all_tokens.len().saturating_sub(self.settings.repeat_last_n);

            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                self.settings.repeat_penalty,
                &self.all_tokens[start_at..],
            )
        }
    }
}

impl<'session, 'model, M: Model> Drop for Generation<'session, 'model, M> {
    fn drop(&mut self) { self.session.tos.clear(); }
}