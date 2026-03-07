//! # BETA!!! NOT WORKING!!!

use tokenizers::Tokenizer;
use crate::{Error, Role};

#[derive(Clone)]
pub(crate) struct ChatTemplate {
    tokenizer: Tokenizer,
}

impl ChatTemplate {
    pub fn new(
        tokenizer: Tokenizer,
    ) -> Result<Self, Error> {
        Ok(Self { tokenizer })
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    pub fn fmt_prompt(&self, prompt: &str, role: Role) -> Result<Vec<u32>, Error> {
        Ok(vec![])
    }

    pub fn assistant_start_template(&self) -> Vec<u32> { // FIX!!! think mode
        vec![]
    }

    pub fn eos_token(&self) -> u32 {
        0
    }
}