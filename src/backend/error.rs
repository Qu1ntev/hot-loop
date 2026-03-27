use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error(transparent)]
    Candle(#[from] candle_core::Error),

    #[error(transparent)]
    Tokenizers(#[from] tokenizers::Error),

    #[error("UnwrapNone: {0}")]
    MissingValue(String),
}