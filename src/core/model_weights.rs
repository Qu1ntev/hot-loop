// use std::io::{Read, Seek};
use candle_core::{Error, Tensor};
use candle_nn::kv_cache::ConcatKvCache;
use tokenizers::Tokenizer;

pub trait ModelWeights {
    fn forward(&self, input: &Tensor, offset: usize, kv_cache: &mut Vec<ConcatKvCache>) -> Result<Tensor, Error>;

    fn create_kv_cache(&self) -> Vec<ConcatKvCache>;

    fn tokenizer(&self) -> &Tokenizer;
}