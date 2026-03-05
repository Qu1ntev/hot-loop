use candle_core::{Device, Result as CandleResult, Tensor};
use crate::utils::kv_cache::ConcatKvCache;
use crate::Error;
use tokenizers::Tokenizer;

pub(crate) type KvCache = ConcatKvCache;

#[derive(Clone, Copy)]
pub enum Role {
    System,
    User,
    Assistant,
}

pub trait ModelWeights {
    fn forward(&self, input: &Tensor, offset: usize, kv_cache: &mut Vec<KvCache>) -> CandleResult<Tensor>;

    fn create_kv_cache(&self) -> Vec<KvCache>;

    fn tokenizer(&self) -> &Tokenizer;

    fn current_device(&self) -> &Device;

    fn fmt_prompt(&self, prompt: &str, role: Role) -> Result<Vec<u32>, Error>;
    fn assistant_start_template(&self) -> Vec<u32>;
    fn eos_token(&self) -> u32;
}

// ADD extend_from_history

pub trait Model: ModelWeights {}

impl<T: ModelWeights> Model for T {}