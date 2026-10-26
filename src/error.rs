//! 统一错误类型。库代码禁止 unwrap/expect 吞错，一律经 Result 上抛。

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("decode: {0}")]
    Decode(String),
    #[error("model: {0}")]
    Model(String),
    #[error("inference: {0}")]
    Inference(String),
    #[error("frontend: {0}")]
    Frontend(String),
    #[error("not found: {0}")]
    NotFound(String),
    #[error("invalid input: {0}")]
    InvalidInput(String),
}

pub type Result<T> = std::result::Result<T, Error>;
