use std::ops::Deref;
use candle_core::{Device, Result as CandleResult, Tensor};
use crate::{session::Session, Error};
use crate::utils::kv_cache::KvCache;
use crate::session::history::Role;
use tokenizers::Tokenizer;
use std::sync::Arc;

#[doc(hidden)]
pub trait ModelWeights {
    fn forward(&self, input: &Tensor, offset: usize, kv_cache: &mut KvCache) -> CandleResult<Tensor>;

    fn layers_len(&self) -> usize;

    fn tokenizer(&self) -> &Tokenizer;

    fn device(&self) -> &Device;

    fn fmt_prompt(&self, prompt: &str, role: Role) -> Result<Vec<u32>, Error>;
    fn assistant_start_template(&self) -> Vec<u32>;
    fn eos_token(&self) -> u32;
}

// impl<M: Model> ModelWeights for M {
//     fn forward(&self, input: &Tensor, offset: usize, kv_cache: &mut KvCache) -> CandleResult<Tensor> {
//         self.forward(input, offset, kv_cache)
//     }
//
//     fn layers_len(&self) -> usize {
//         self.layers_len()
//     }
//
//     fn tokenizer(&self) -> &Tokenizer {
//         self.tokenizer()
//     }
//
//     fn device(&self) -> &Device {
//         self.device()
//     }
//
//     fn fmt_prompt(&self, prompt: &str, role: Role) -> Result<Vec<u32>, Error> {
//         self.fmt_prompt(prompt, role)
//     }
//     fn assistant_start_template(&self) -> Vec<u32> {
//         self.assistant_start_template()
//     }
//     fn eos_token(&self) -> u32 {
//         self.eos_token()
//     }
// }

pub trait Model: ModelWeights {
    fn new_session(&self) -> Session<Self>
    where
        Self: Sized;
}

impl<M: Model> Model for Arc<M> {
    fn new_session(&self) -> Session<Self> {
        Session::new(Arc::clone(self))
    }
}

impl<M: Model> Model for &M {
    fn new_session(&self) -> Session<Self> {
        Session::new(*self)
    }
}

impl<M: Deref<Target: Model>> ModelWeights for M {
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

    fn fmt_prompt(&self, prompt: &str, role: Role) -> Result<Vec<u32>, Error> {
        self.deref().fmt_prompt(prompt, role)
    }
    fn assistant_start_template(&self) -> Vec<u32> {
        self.deref().assistant_start_template()
    }
    fn eos_token(&self) -> u32 {
        self.deref().eos_token()
    }
}

// impl<M: ModelWeights + Clone> Model for M {
//     fn new_session(&self) -> Session<M> {
//         Session::new(self.clone())
//     }
// }