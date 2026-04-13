use candle_core::Tensor;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use super::Generation;
use crate::settings::{Settings, Seed};
use crate::Error;
use crate::Model;
use crate::session::history::Message;
use crate::utils::kv_cache::KvCache;
use super::generation::Phase;
use crate::models::models_core::model::ChatTemplate;

#[non_exhaustive]
pub struct Session<M: Model> {
    model: M, // read only
    settings: Settings,
    kv_cache: KvCache,
    cached_tokens: Vec<u32>
}

impl<M: Model> Session<M> {
    pub fn new(model: M) -> Self {
        let settings = Settings::default();
        let cached_tokens = Vec::new();

        let layers_len = model.layers_len();
        let kv_cache = KvCache::new(layers_len, 2);
        
        Self {
            model,
            settings,
            kv_cache,
            cached_tokens,
        }
    }

    pub fn generate(&mut self, history: &[Message]) -> Result<Generation<'_, M>, Error> {
        if history.is_empty() {
            return Err(Error::MissingValue("History is empty".into()));
        }

        let tokens = self.model
            .chat_format()
            .fmt_history(self.model.tokenizer(), history, true)?;
        
        let mask = self.history_mask(&tokens);

        let new_tokens = tokens[mask..].to_vec();

        let phase = match new_tokens.is_empty() {
            true => {
                let token = *tokens.last()
                    .ok_or_else(|| Error::MissingValue("cached_tokens last is empty".into()))?;

                let satur_mask = mask.saturating_sub(1);

                self.kv_cache.truncate(satur_mask)?;
                self.cached_tokens.truncate(satur_mask);

                Phase::Decode(token)
            },
            false => {
                self.kv_cache.truncate(mask)?;
                self.cached_tokens.truncate(mask);

                Phase::Prefill(new_tokens)
            }
        };

        let sampling = self.sampling();
        let seed = self.seed();

        let logits_processor =
            LogitsProcessor::from_sampling(seed, sampling);

        Ok(Generation::new(
            &self.model,
            phase,
            &mut self.kv_cache,
            &mut self.cached_tokens,
            logits_processor,
            self.settings,
        ))
    }

    fn history_mask(&self, tokens: &[u32]) -> usize {
        self.cached_tokens.iter()
            .zip(tokens.iter())
            .take_while(|(a, b)| a == b)
            .count()
    }
    
    fn sampling(&self) -> Sampling {
        let temperature = self.settings.temperature;

        if temperature <= 0. {
            Sampling::ArgMax
        } else {
            match (self.settings.top_k, self.settings.top_p) {
                (None, None) => Sampling::All { temperature },
                (Some(k), None) => Sampling::TopK { k, temperature },
                (None, Some(p)) => Sampling::TopP { p, temperature },
                (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
            }
        }
    }

    fn seed(&self) -> u64 {
        match self.settings.seed {
            Seed::Custom(seed) => seed,
            Seed::Random => rand::random()
        }
    }

    pub fn with_settings(mut self, settings: Settings) -> Self {
        self.settings = settings;
        self
    }

    pub fn set_settings(&mut self, settings: Settings) {
        self.settings = settings;
    }

    pub fn clear_cache(&mut self) {
        self.kv_cache.clear();
        self.cached_tokens.clear();
    }

    pub fn context(&self) -> usize {
        self.kv_cache.current_pos()
    }

    pub fn warmup(&mut self, history: &[Message]) -> Result<(), Error> {
        if history.is_empty() {
            return Ok(());
        }

        let tokens = self.model
            .chat_format()
            .fmt_history(self.model.tokenizer(), history, false)?;
        
        let mask = self.history_mask(&tokens);

        let new_tokens = tokens[mask..].to_vec();

        match new_tokens.is_empty() {
            true =>
                return Ok(()),

            false => {
                self.kv_cache.truncate(mask)?;
                self.cached_tokens.truncate(mask);

                let input = Tensor::new(new_tokens.as_slice(), self.model.device())?.unsqueeze(0)?;
                let current_pos = self.kv_cache.current_pos();

                let _ = self.model.forward(&input, current_pos, &mut self.kv_cache)?;

                self.cached_tokens.extend_from_slice(&new_tokens);
            }
        }

        Ok(())
    }
}

#[cfg(feature = "dev")]
impl<M: Model> Session<M> {
    pub fn cached_tokens_str(&self) -> Result<String, Error> {
        Ok(self.model.tokenizer().decode(&self.cached_tokens, false)?)
    }

    pub fn cached_tokens(&self) -> &[u32] {
        &self.cached_tokens
    }
}