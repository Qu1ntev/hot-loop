use candle_core::{Device, Result as CandleResult, Tensor};
use crate::{Error, session::Session, KvCache};
use tokenizers::Tokenizer;

#[derive(Clone, Copy)]
pub enum Role {
    System,
    User,
    Assistant,
}

pub trait ModelWeights {
    fn forward(&self, input: &Tensor, offset: usize, kv_cache: &mut Vec<KvCache>) -> CandleResult<Tensor>;
    
    fn layers_len(&self) -> usize;
    
    fn create_kv_cache(&self) -> Vec<KvCache> {
        let layers_len = self.layers_len();
        
        let mut kv_cache = Vec::with_capacity(layers_len);

        for _ in 0..layers_len {
            kv_cache.push(KvCache::new(2));
        }

        kv_cache
    }

    fn tokenizer(&self) -> &Tokenizer;

    fn current_device(&self) -> &Device;

    fn fmt_prompt(&self, prompt: &str, role: Role) -> Result<Vec<u32>, Error>;
    fn assistant_start_template(&self) -> Vec<u32>;
    fn eos_token(&self) -> u32;
}

// ADD extend_from_history

pub trait Model: ModelWeights {
    fn new_session(&self) -> Session<'_, Self>
    where
        Self: Sized
    {
        Session::new(self)
    }
}

impl<T: ModelWeights> Model for T {}