use candle_transformers::generation::{LogitsProcessor, Sampling};
use super::Generation;
use super::history::Message;
use crate::Error;
use crate::Model;
use crate::settings::{Settings, Seed};
use crate::utils::kv_cache::KvCache;
use crate::utils::token_output_stream::TokenOutputStream;

#[non_exhaustive]
pub struct Session<'model, M: Model> {
    model: &'model M, // read only
    settings: Settings,
    kv_cache: Vec<KvCache>,
    tos: TokenOutputStream<'model>,
    cached_tokens: Vec<u32>
}

impl<'model, M: Model> Session<'model, M> {
    pub(crate) fn new(model: &'model M) -> Self {
        let settings = Settings::default();
        let kv_cache = model.create_kv_cache();
        let tos = TokenOutputStream::new(model.tokenizer());
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
        let tokens = self.message_tokens(messages)?;

        let cached_len = self.cached_tokens
            .iter()
            .zip(tokens.iter())
            .take_while(|(a, b)| a == b)
            .count();

        println!("cached len: {}", cached_len);

        self.truncate_cache(cached_len)?;
        self.cached_tokens = tokens.clone();

        println!("tokens len: {}", tokens.len());

        let mut tokens = tokens[cached_len..].to_vec();

        let assistant_start_tokens = self.model.assistant_start_template();
        tokens.extend_from_slice(&assistant_start_tokens);

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

        Ok(Generation {
            model: self.model,
            index: 0,
            next_token: 0,
            tokens,
            all_tokens: Vec::new(),
            parameters: self.settings,
            device: self.model.current_device(),
            eos_tokens: self.model.eos_tokens(),
            logits_processor,
            tos: &mut self.tos,
            kv_cache: &mut self.kv_cache,
            cached_tokens: &mut self.cached_tokens
        })
    }

    pub fn with_settings(mut self, settings: Settings) -> Self {
        self.settings = settings;
        self
    }

    pub fn set_settings(&mut self, settings: Settings) {
        self.settings = settings;
    }

    pub fn warmup(&mut self, _messages: &[Message]) -> Result<(), Error> {
        // warmup kv cache, can be used, for example, for a system prompt
        Ok(())
    }

    fn truncate_cache(&mut self, index: usize) -> Result<(), Error> {
        for cache in &mut self.kv_cache {
            cache.truncate(index)?;
        }
        Ok(())
    }

    pub fn clear_cache(&mut self) {
        self.cached_tokens.clear();

        for cache in &mut self.kv_cache {
            cache.clear();
        }
    }
    
    fn message_tokens(&self, messages: &[Message]) -> Result<Vec<u32>, Error> {
        let mut tokens = Vec::new();
        
        for message in messages {
            let tk = self.model.fmt_prompt(
                message.role,
                &message.text
            )?;
            println!("{:?} tokens len: {}", message.role, tk.len());
            tokens.extend_from_slice(&tk);
        }
        
        Ok(tokens)
    }
}