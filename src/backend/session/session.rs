use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_core::Tensor;
use super::Generation;
use crate::{
    Error, ModelWeights, KvCache,
    settings::{Settings, Seed},
    Role
};
use crate::utils::token_output_stream::TokenOutputStream;

#[non_exhaustive]
#[derive(Clone)]
pub struct Session<'a, M: ModelWeights> {
    model: &'a M, // read only
    settings: Settings,
    kv_cache: Vec<KvCache>,
    tos: TokenOutputStream<'a>
}

impl<'a, M: ModelWeights> Session<'a, M> {
    pub(crate) fn new(model: &'a M) -> Self {
        let settings = Settings::default();
        let kv_cache = model.create_kv_cache();
        let tos = TokenOutputStream::new(model.tokenizer());
        
        Self {
            model,
            settings,
            kv_cache,
            tos
        }
    }

    pub fn generate(&mut self, prompt: &str) -> Result<Generation<'_, 'a, M>, Error> {
        let user_tokens = self.model.fmt_prompt(prompt, Role::User)?;
        let assistant_start_tokens = self.model.assistant_start_template();

        let mut tokens = Vec::with_capacity(
            user_tokens.len() + assistant_start_tokens.len()
        );

        tokens.extend_from_slice(&user_tokens);
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
            eos_token: self.model.eos_token(),
            logits_processor,
            tos: &mut self.tos,
            kv_cache: &mut self.kv_cache
        })
    }

    pub fn with_settings(mut self, settings: Settings) -> Self {
        self.settings = settings;
        self
    }

    pub fn set_settings(&mut self, settings: Settings) {
        self.settings = settings;
    }

    pub fn with_system_prompt(mut self, system_prompt: &str) -> Result<Self, Error> {
        self.set_system_prompt_and_clear_history(system_prompt)?;
        Ok(self)
    }

    pub fn set_system_prompt_and_clear_history(&mut self, system_prompt: &str) -> Result<(), Error> {
        self.reset_all_cache();

        let sys_tokens = self.model.fmt_prompt(system_prompt, Role::System)?;

        let input = Tensor::new(sys_tokens, self.model.current_device())?.unsqueeze(0)?;
        let _ = self.model.forward(&input, 0, &mut self.kv_cache)?;

        self.set_cache_rollback();

        Ok(())
    }

    pub fn clear_history(&mut self) {
        self.cache_rollback();
    }

    pub fn clear_system_prompt_and_history(&mut self) {
        self.reset_all_cache();
    }

    fn set_cache_rollback(&mut self) {
        for cache in &mut self.kv_cache {
            cache.set_rollback();
        }
    }

    fn cache_rollback(&mut self) {
        for cache in &mut self.kv_cache {
            cache.rollback();
        }
    }

    fn reset_all_cache(&mut self) {
        for cache in &mut self.kv_cache {
            cache.reset_all();
        }
    }
}