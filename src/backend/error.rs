use candle_transformers::models::mimi::candle;
use thiserror::Error;
use tokenizers::tokenizer;

#[derive(Error, Debug)]
pub enum Error {
    #[error(transparent)]
    Candle(#[from] candle::Error),

    #[error(transparent)]
    Tokenizers(#[from] tokenizer::Error),

    #[error("is_none")]
    None
}

// ADD KIND

// pub enum ErrorKind {
//
// }
//
// impl Error {
//     pub fn kind(&self) -> ErrorKind {
//         match self {
//
//         }
//     }
// }