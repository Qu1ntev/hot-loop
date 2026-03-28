use candle_transformers::generation::{LogitsProcessor, Sampling};
use super::Generation;
use crate::Error;
use crate::Model;
use crate::session::history::Message;
use crate::settings::{Settings, Seed};
use crate::utils::kv_cache::KvCache;
use crate::utils::token_output_stream::TokenOutputStream;
use crate::ChatFormat;

// struct CachedTokens {
//
// }

#[non_exhaustive]
pub struct Session<'model, M: Model> {
    pub(crate) model: &'model M, // read only
    pub(crate) settings: Settings,
    pub(crate) kv_cache: KvCache,
    pub(crate) tos: TokenOutputStream<'model>,
    pub(crate) cached_tokens: Vec<u32>
}

impl<'model, M: Model> Session<'model, M> {
    pub(crate) fn new(model: &'model M) -> Self {
        let settings = Settings::default();
        
        let layers_len = model.layers_len();
        let kv_cache = KvCache::new(layers_len, 2);
        
        let tos = TokenOutputStream::new(model.chat_format().tokenizer());
        let cached_tokens = Vec::new();
        
        Self {
            model,
            settings,
            kv_cache,
            tos,
            cached_tokens
        }
    }

    pub fn generate(&mut self, messages: &[Message]) -> Result<Generation<'_, 'model, M>, Error> {
        let tokens = self.model.chat_format().format_for_gen(&messages, true)?;
        let cached_len = self.history_mask(&tokens);

        self.kv_cache.truncate(cached_len)?;
        self.cached_tokens = tokens.clone();

        let tokens = tokens[cached_len..].to_vec();

        let logits_processor = {
            let temperature = self.settings.temperature;
            let sampling = if temperature <= 0. {
                Sampling::ArgMax
            } else {
                match (self.settings.top_k, self.settings.top_p) {
                    (None, None) => Sampling::All { temperature },
                    (Some(k), None) => Sampling::TopK { k, temperature },
                    (None, Some(p)) => Sampling::TopP { p, temperature },
                    (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
                }
            };

            let seed = match self.settings.seed {
                Seed::Custom(seed) => seed,
                Seed::Default => 299792458 // temporary
            };

            LogitsProcessor::from_sampling(seed, sampling)
        };

        Ok(Generation::new(
            self,
            tokens,
            logits_processor,
            self.settings
        ))
    }

    fn history_mask(&self, tokens: &[u32]) -> usize {
        self.cached_tokens.iter()
            .zip(tokens.iter())
            .take_while(|(a, b)| a == b)
            .count()
    }

    pub fn with_settings(mut self, settings: Settings) -> Self {
        self.settings = settings;
        self
    }

    pub fn set_settings(&mut self, settings: Settings) {
        self.settings = settings;
    }

    // pub fn warmup(&mut self, _messages: &[Message]) -> Result<(), Error> {
    //     self.clear_cache();
    //     // warmup kv cache, can be used, for example, for a system prompt
    //     Ok(())
    // }

    pub fn clear_cache(&mut self) {
        self.cached_tokens.clear();
        self.kv_cache.clear();
    }
}