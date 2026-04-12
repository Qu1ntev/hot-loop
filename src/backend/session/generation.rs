use candle_core::Tensor;
use candle_core::Result as CandleResult;
use candle_transformers::generation::LogitsProcessor;
use crate::settings::Settings;
use crate::Error;
use crate::Model;
use crate::utils::token_output_stream::TokenOutputStream;
use crate::utils::kv_cache::KvCache;

pub(crate) enum Phase {
    Prefill(Vec<u32>),
    Decode(u32),
}

#[non_exhaustive]
pub struct Generation<'session, M: Model> {
    model: &'session M,
    kv_cache: &'session mut KvCache,
    index: usize,
    phase: Phase,
    cached_tokens: &'session mut Vec<u32>,
    response_tokens: Vec<u32>,
    settings: Settings,
    logits_processor: LogitsProcessor,
    tos: TokenOutputStream,
}

impl<'session, M: Model> Generation<'session, M> {
    pub(crate) fn new(
        model: &'session M,
        phase: Phase,
        kv_cache: &'session mut KvCache,
        cached_tokens: &'session mut Vec<u32>,
        logits_processor: LogitsProcessor,
        settings: Settings,
    ) -> Self {
        Self {
            model,
            kv_cache,
            phase,
            cached_tokens,
            logits_processor,
            settings,
            index: 0,
            response_tokens: Vec::new(),
            tos: TokenOutputStream::new(),
        }
    }

    pub fn next_chunk(&mut self) -> Result<Option<String>, Error> {
        loop {
            if self.is_len_limit() {
                return Ok(None);
            }

            if let Phase::Decode(next_token) = self.phase {
                self.response_tokens.push(next_token);
                self.cached_tokens.push(next_token);
            }

            let input = self.input_token()?;
            let logits = self.model_infer(input)?;
            let logits = self.apply_repeat_penalty(logits)?;

            let next_token = self.logits_processor.sample(&logits)?;
            self.phase = Phase::Decode(next_token);

            self.index += 1;

            if self.is_model_return(next_token) {
                return Ok(None);
            }

            if let Some(chunk) = self.has_chunk(next_token)? {
                return Ok(Some(chunk))
            }
        }
    }

    fn is_len_limit(&self) -> bool {
        self.settings.sample_len <= self.index
    }

    fn is_model_return(&self, next_token: u32) -> bool {
        next_token == self.model.eos_token()
    }

    fn has_chunk(&mut self, next_token: u32) -> CandleResult<Option<String>> {
        self.tos.next_token(self.model.tokenizer(), next_token)
    }

    fn model_infer(&mut self, input: Tensor) -> CandleResult<Tensor> {
        let input = input.unsqueeze(0)?;

        let current_pos = self.kv_cache.current_pos();
        let logits = self.model.forward(&input, current_pos, &mut self.kv_cache)?;

        logits.squeeze(0)
    }

    fn input_token(&mut self) -> CandleResult<Tensor> {
        match &self.phase {
            Phase::Prefill(tokens) =>
                Tensor::new(tokens.as_slice(), self.model.device()),
            
            Phase::Decode(token) =>
                Tensor::new(&[*token], self.model.device()),
        }
    }

    fn apply_repeat_penalty(&self, logits: Tensor) -> CandleResult<Tensor> {
        if self.settings.repeat_penalty == 1. {
            Ok(logits)
        } else {
            let start_at = self.response_tokens.len().saturating_sub(self.settings.repeat_last_n);
            
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                self.settings.repeat_penalty,
                &self.response_tokens[start_at..],
            )
        }
    }
}