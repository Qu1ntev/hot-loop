use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_core::Tensor;
use crate::{Error, Generation, Settings, ModelWeights, KvCache};
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
        // self.model.clear_kv_cache(); // FIX!!! history

        let prompt_str = format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"); // FIX!!! history

        let tokens = self.model.tokenizer() // FIX!!! special tokens
            .encode(prompt_str, true)?;

        let tokens = tokens.get_ids().to_vec();

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
            LogitsProcessor::from_sampling(299792458, sampling) // FIX!!! seed hard-code
        };

        let eos_token = *self.model.tokenizer() // FIX!!! const or parse from model
            .get_vocab(true)
            .get("<|im_end|>")
            .ok_or(Error::None)?;

        Ok(Generation {
            model: self.model,
            index: 0,
            next_token: 0,
            tokens,
            all_tokens: Vec::new(),
            parameters: self.settings,
            device: self.model.current_device(),
            eos_token,
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
        self.set_system_prompt(system_prompt)?;
        Ok(self)
    }

    pub fn set_system_prompt(&mut self, system_prompt: &str) -> Result<(), Error> {
        self.clear_history();

        let sys_prompt_str = format!("<|im_start|>system\n{system_prompt}<|im_end|>\n");
        let tokens = self.model.tokenizer().encode(sys_prompt_str, true)?;
        let token_ids = tokens.get_ids();

        let input = Tensor::new(token_ids, self.model.current_device())?.unsqueeze(0)?;
        let _ = self.model.forward(&input, 0, &mut self.kv_cache)?;

        self.set_rollback();

        Ok(())
    }

    pub fn clear_history(&mut self) {
        self.rollback();
    }

    pub fn clear_system_prompt_and_history(&mut self) {
        self.reset_all();
    }

    fn set_rollback(&mut self) {
        for cache in &mut self.kv_cache {
            cache.set_rollback();
        }
    }

    fn rollback(&mut self) {
        for cache in &mut self.kv_cache {
            cache.rollback();
        }
    }

    fn reset_all(&mut self) {
        for cache in &mut self.kv_cache {
            cache.reset_all();
        }
    }
}