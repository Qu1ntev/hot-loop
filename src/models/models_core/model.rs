use std::io::{Read, Seek};
use std::ops::Deref;
use candle_core::{DType, Device, Result as CandleResult, Tensor};
use crate::{session::Session, Error};
use crate::utils::kv_cache::KvCache;
use crate::session::history::Message;
use tokenizers::Tokenizer;

#[doc(hidden)]
pub trait ModelWeights {
    fn forward(&self, input: &Tensor, offset: usize, kv_cache: &mut KvCache) -> CandleResult<Tensor>;

    fn layers_len(&self) -> usize;

    fn tokenizer(&self) -> &Tokenizer;

    fn device(&self) -> &Device;

    fn chat_format(&self) -> &impl ChatTemplate;
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

    fn chat_format(&self) -> &impl ChatTemplate {
        self.deref().chat_format()
    }
}

pub trait Loadable: Sized {
    fn load<R: Read + Seek>(
        model: R,
        tokenizer: Option<Vec<u8>>,
        device: Device,
        dtype: DType,
    ) -> Result<Self, Error>;
}

pub trait ChatTemplate {
    fn fmt_history(
        &self,
        tokenizer: &Tokenizer,
        history: &[Message],
        add_start: bool,
    ) -> Result<Vec<u32>, Error>;

    fn eos_token(&self) -> u32;
}