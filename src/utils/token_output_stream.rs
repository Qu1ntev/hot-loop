use candle_core::Result;
use tokenizers::Tokenizer;

const REPL: char = '\u{FFFD}';

#[derive(Clone)]
pub struct TokenOutputStream<'a> {
    tokenizer: &'a Tokenizer,
    tokens: Vec<u32>,
    prev_index: usize,
    current_index: usize,
}

impl<'a> TokenOutputStream<'a> {
    pub fn new(tokenizer: &'a Tokenizer) -> Self {
        Self {
            tokenizer,
            tokens: Vec::new(),
            prev_index: 0,
            current_index: 0,
        }
    }

    pub fn next_token(&mut self, token: u32) -> Result<Option<String>> {
        let prev_text = self.get_prev_text()?;
        self.tokens.push(token);
        let text = self.decode(&self.tokens[self.prev_index..])?;

        if text.len() > prev_text.len() && !text.ends_with(REPL) {
            let result = text.split_at(prev_text.len()).1.to_string();

            self.prev_index = self.current_index;
            self.current_index = self.tokens.len();

            Ok(Some(result))

        } else {
            Ok(None)
        }
    }

    pub fn clear(&mut self) {
        self.tokens.clear();
        self.prev_index = 0;
        self.current_index = 0;
    }

    fn decode(&self, tokens: &[u32]) -> Result<String> {
        match self.tokenizer.decode(tokens, true) {
            Ok(str) => Ok(str),
            Err(err) => candle_core::bail!("cannot decode: {err}"),
        }
    }

    fn get_prev_text(&self) -> Result<String> {
        if self.tokens.is_empty() {
            Ok(String::new())
        } else {
            let tokens = &self.tokens[self.prev_index..self.current_index];
            self.decode(tokens)
        }
    }
}