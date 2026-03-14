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
        let prompt = "\
        <bos><start_of_turn>user\n\
        Ты Умный Ассистент\n\n\
        привет<end_of_turn>\n\
        <start_of_turn>model\n\
        ";
        let tokens = self.tokenizer.encode(prompt, true)?;
        Ok(tokens.get_ids().to_vec())
    }

    pub fn assistant_start_template(&self) -> Vec<u32> { // FIX!!! think mode
        vec![]
    }

    pub fn eos_token(&self) -> u32 {
        let token = self.tokenizer.encode("<end_of_turn>", true).unwrap();
        token.get_ids().get(0).unwrap().clone()
    }
}