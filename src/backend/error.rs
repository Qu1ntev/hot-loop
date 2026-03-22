use thiserror::Error;
use tokenizers::tokenizer;

#[derive(Error, Debug)]
pub enum Error {
    #[error(transparent)]
    Candle(#[from] candle_core::Error),

    #[error(transparent)]
    Tokenizers(#[from] tokenizer::Error),

    #[error("UnwrapNone: {0}")]
    UnwrapNone(String),
}