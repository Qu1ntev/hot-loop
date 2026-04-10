use candle_transformers::generation::{LogitsProcessor, Sampling};
use super::Generation;
use crate::settings::{Settings, Seed};
use crate::Error;
use crate::Model;
use crate::session::history::Message;
use crate::utils::kv_cache::KvCache;
use crate::utils::token_output_stream::TokenOutputStream;

#[non_exhaustive]
pub struct Session<M: Model> {
    model: M, // read only
    settings: Settings,
    kv_cache: KvCache,
    tos: TokenOutputStream,
}

impl<M: Model> Session<M> {
    pub fn new(model: M) -> Self {
        let settings = Settings::default();

        let layers_len = model.layers_len();
        let kv_cache = KvCache::new(layers_len, 2);

        let tos = TokenOutputStream::new();
        
        Self {
            model,
            settings,
            kv_cache,
            tos,
        }
    }

    pub fn generate(&mut self, history: &[Message]) -> Result<Generation<'_, M>, Error> {
        let tokens = self.model.fmt_history(history)?;

        let sampling = self.sampling();
        let seed = self.seed();

        let logits_processor =
            LogitsProcessor::from_sampling(seed, sampling);

        Ok(Generation::new(
            &self.model,
            tokens,
            logits_processor,
            self.settings,
            &mut self.tos,
            &mut self.kv_cache,
        ))
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
    }
}