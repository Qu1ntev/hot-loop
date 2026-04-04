use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_core::Tensor;
use super::Generation;
use crate::settings::{Settings, Seed};
use crate::Error;
use crate::Model;
use crate::session::history::Role;
use crate::utils::kv_cache::KvCache;
use crate::utils::token_output_stream::TokenOutputStream;

#[non_exhaustive]
pub struct Session<M: Model> {
    model: M, // read only
    settings: Settings,
    kv_cache: KvCache,
    tos: TokenOutputStream,
    system_prompt_pos: Option<usize>,
}

impl<M: Model> Session<M> {
    pub(crate) fn new(model: M) -> Self {
        let settings = Settings::default();

        let layers_len = model.layers_len();
        let kv_cache = KvCache::new(layers_len, 2);

        let tos = TokenOutputStream::new(model.tokenizer());
        
        Self {
            model,
            settings,
            kv_cache,
            tos,
            system_prompt_pos: None,
        }
    }

    pub fn generate(&mut self, prompt: &str) -> Result<Generation<'_, M>, Error> {
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
                Seed::Random => rand::random()
            };

            LogitsProcessor::from_sampling(seed, sampling)
        };

        Ok(Generation::new(
            &self.model,
            tokens,
            logits_processor,
            self.settings,
            &mut self.tos,
            &mut self.kv_cache,
            self.model.eos_token(),
        ))
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
        self.kv_cache.clear();

        let sys_tokens = self.model.fmt_prompt(system_prompt, Role::System)?;
        let input = Tensor::new(sys_tokens, self.model.device())?.unsqueeze(0)?;
        let _ = self.model.forward(&input, 0, &mut self.kv_cache)?;

        let current_pos = self.kv_cache.current_pos();

        self.system_prompt_pos = Some(current_pos);

        Ok(())
    }

    pub fn clear_history(&mut self) -> Result<(), Error> {
        match self.system_prompt_pos {
            Some(pos) => self.kv_cache.truncate(pos)?,
            None => self.kv_cache.clear()
        }
        Ok(())
    }

    pub fn clear_system_prompt_and_history(&mut self) {
        self.kv_cache.clear();
    }
}