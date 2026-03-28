use candle_core::{Device, Result as CandleResult, Tensor};
use crate::{session::Session, Error};
use crate::utils::kv_cache::KvCache;
use tokenizers::Tokenizer;
use crate::session::history::Message;

#[doc(hidden)]
pub trait ModelWeights {
    type ChatFormat: ChatFormat;
    
    fn forward(&self, input: &Tensor, offset: usize, kv_cache: &mut KvCache) -> CandleResult<Tensor>;

    fn layers_len(&self) -> usize;

    fn device(&self) -> &Device;

    fn chat_format(&self) -> &Self::ChatFormat;
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

#[doc(hidden)]
pub trait ChatFormat {
    fn format_for_gen(&self, messages: &[Message], model_start: bool) -> Result<Vec<u32>, Error>;
    fn eos_tokens(&self) -> &[u32];
    fn tokenizer(&self) -> &Tokenizer;
}