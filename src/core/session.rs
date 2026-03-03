use candle_core::Device;
use candle_nn::kv_cache::{ConcatKvCache};
use candle_transformers::generation::{LogitsProcessor, Sampling};
// use tokenizers::Tokenizer;
use crate::{Error, Generation, Settings, ModelWeights};
// use crate::models::qwen3::qwen3_weights::Qwen3Weights;
use crate::utils::token_output_stream::TokenOutputStream;

pub struct Session<'a, M: ModelWeights> {
    model: &'a M,
    settings: Settings,
    kv_cache: Vec<ConcatKvCache> // FIX!!! cache type hard-code
}

impl<'a, M: ModelWeights> Session<'a, M> {
    pub fn new(model: &'a M) -> Self {
        let settings = Settings::default();
        let kv_cache = model.create_kv_cache();
        
        Self {
            model,
            settings,
            kv_cache
        }
    }

    pub fn generate(&mut self, prompt: &str) -> Result<Generation<'_, M>, Error> {
        let tos = TokenOutputStream::new(self.model.tokenizer().clone());

        // self.model.clear_kv_cache(); // FIX!!! history

        let prompt_str = format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"); // FIX!!! history

        let tokens = tos // FIX!!! special tokens
            .tokenizer()
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

        let eos_token = *tos.tokenizer() // FIX!!! const or parse from model
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
            device: &Device::Cpu, // &self.device, // FIX!!! device hard-code
            eos_token,
            logits_processor,
            tos,
            kv_cache: &mut self.kv_cache
        })
    }
}