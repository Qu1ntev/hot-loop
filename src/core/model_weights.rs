use candle_core::{Device, Error, Tensor};
// use candle_nn::kv_cache::ConcatKvCache;
use crate::utils::kv_cache::ConcatKvCache;
use tokenizers::Tokenizer;

pub(crate) type KvCache = ConcatKvCache;

pub trait ModelWeights {
    fn forward(&self, input: &Tensor, offset: usize, kv_cache: &mut Vec<KvCache>) -> Result<Tensor, Error>;

    fn create_kv_cache(&self) -> Vec<KvCache>;

    fn tokenizer(&self) -> &Tokenizer;

    fn current_device(&self) -> &Device;
}

// fn chat_template_refactor(&self, history: History) -> String

pub trait Model: ModelWeights {}

impl<T: ModelWeights> Model for T {}