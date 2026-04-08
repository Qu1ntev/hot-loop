use candle_core::Tensor;
use candle_core::Result as CandleResult;
use candle_transformers::generation::LogitsProcessor;
use crate::settings::Settings;
use crate::Error;
use crate::Model;
use crate::utils::token_output_stream::TokenOutputStream;
use crate::utils::kv_cache::KvCache;

#[non_exhaustive]
pub struct Generation<'session, M: Model> {
    model: &'session M,
    index: usize,
    next_token: u32,
    tokens_prefill: Option<Vec<u32>>,
    all_tokens: Vec<u32>,
    settings: Settings,
    logits_processor: LogitsProcessor,
    tos: &'session mut TokenOutputStream,
    kv_cache: &'session mut KvCache
}

impl<'session, M: Model> Generation<'session, M> {
    pub(crate) fn new(
        model: &'session M,
        tokens_prefill: Vec<u32>,
        logits_processor: LogitsProcessor,
        settings: Settings,
        tos: &'session mut TokenOutputStream,
        kv_cache: &'session mut KvCache,
    ) -> Self {
        Self {
            model,
            index: 0,
            next_token: 0,
            all_tokens: Vec::new(),
            tokens_prefill: Some(tokens_prefill),
            logits_processor,
            settings,
            tos,
            kv_cache,
        }
    }

    pub fn next_chunk(&mut self) -> Result<Option<String>, Error> {
        loop {
            if self.is_len_limit() {
                return Ok(None);
            }

            let logits = self.model_infer()?;

            let logits = self.apply_repeat_penalty(logits)?;

            self.next_token = self.logits_processor.sample(&logits)?;
            self.all_tokens.push(self.next_token);

            self.index += 1;

            if self.is_model_return() {
                return Ok(None);
            }

            if let Some(chunk) = self.has_chunk()? {
                return Ok(Some(chunk))
            }
        }
    }

    fn is_len_limit(&self) -> bool {
        self.settings.sample_len <= self.index
    }

    fn is_model_return(&self) -> bool {
        self.next_token == self.model.eos_token()
    }

    fn has_chunk(&mut self) -> CandleResult<Option<String>> {
        self.tos.next_token(self.model.tokenizer(), self.next_token)
    }

    fn model_infer(&mut self) -> CandleResult<Tensor> {
        let input = if self.index == 0 &&
            let Some(tokens_prefill) = self.tokens_prefill.take() {
            Tensor::new(tokens_prefill.as_slice(), self.model.device())?

        } else {
            Tensor::new(&[self.next_token], self.model.device())?
        };
        let input = input.unsqueeze(0)?;

        let current_pos = self.kv_cache.current_pos();
        let logits = self.model.forward(&input, current_pos, &mut self.kv_cache)?;

        logits.squeeze(0)
    }

    fn apply_repeat_penalty(&self, logits: Tensor) -> CandleResult<Tensor> {
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

impl<'session, M: Model> Drop for Generation<'session, M> {
    fn drop(&mut self) {
        self.tos.clear();
    }
}