use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error(transparent)]
    Candle(#[from] candle_core::Error),

    #[error(transparent)]
    Tokenizers(#[from] tokenizers::Error),

    #[error("UnwrapNone: {0}")]
    UnwrapNone(String),
}

#[derive(Debug)]
pub enum ErrorKind {
    Candle,
    Tokenizer,
    UnwrapNone,
}

impl Error {
    pub fn kind(&self) -> ErrorKind {
        match self {
            Error::Candle(_) => ErrorKind::Candle,
            Error::Tokenizers(_) => ErrorKind::Tokenizer,
            Error::UnwrapNone(_) => ErrorKind::UnwrapNone,
        }
    }
}