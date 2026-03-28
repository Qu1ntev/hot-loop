use candle_core::{Device, Result as CandleResult, Tensor};
use crate::{session::Session, Error};
use crate::utils::kv_cache::KvCache;
use tokenizers::Tokenizer;
use crate::session::history::Role;

pub trait ModelWeights {
    fn forward(&self, input: &Tensor, offset: usize, kv_cache: &mut KvCache) -> CandleResult<Tensor>;

    fn layers_len(&self) -> usize;

    fn tokenizer(&self) -> &Tokenizer;

    fn device(&self) -> &Device;

    fn fmt_prompt(&self, prompt: &str, role: Role) -> Result<Vec<u32>, Error>;
    fn assistant_start_template(&self) -> Vec<u32>;
    fn eos_token(&self) -> u32;
}

pub trait Model: ModelWeights {
    fn new_session(&self) -> Session<'_, Self>
    where
        Self: Sized
    {
        Session::new(self)
    }
}

impl<T: ModelWeights> Model for T {}