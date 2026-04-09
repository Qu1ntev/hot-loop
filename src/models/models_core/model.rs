use std::ops::Deref;
use candle_core::{Device, Result as CandleResult, Tensor};
use crate::{session::Session, Error};
use crate::utils::kv_cache::KvCache;
use tokenizers::Tokenizer;
use crate::session::history::History;

#[doc(hidden)]
pub trait ModelWeights {
    fn forward(&self, input: &Tensor, offset: usize, kv_cache: &mut KvCache) -> CandleResult<Tensor>;

    fn layers_len(&self) -> usize;

    fn tokenizer(&self) -> &Tokenizer;

    fn device(&self) -> &Device;

    fn fmt_history<H: History>(&self, history: H) -> Result<Vec<u32>, Error>;
    
    fn eos_token(&self) -> u32;
}

pub trait Model: ModelWeights + Send + Sync {
    fn new_session(self) -> Session<Self>
    where
        Self: Sized
    {
        Session::new(self)
    }
}

impl<M: ModelWeights + Send + Sync> Model for M {}

impl<M: Deref<Target: ModelWeights>> ModelWeights for M {
    fn forward(&self, input: &Tensor, offset: usize, kv_cache: &mut KvCache) -> CandleResult<Tensor> {
        self.deref().forward(input, offset, kv_cache)
    }

    fn layers_len(&self) -> usize {
        self.deref().layers_len()
    }

    fn tokenizer(&self) -> &Tokenizer {
        self.deref().tokenizer()
    }

    fn device(&self) -> &Device {
        self.deref().device()
    }

    fn fmt_history<H: History>(&self, history: H) -> Result<Vec<u32>, Error> {
        self.deref().fmt_history(history)
    }

    fn eos_token(&self) -> u32 {
        self.deref().eos_token()
    }
}