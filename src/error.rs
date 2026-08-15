// Copyright (c) 2025 xiefujin <490021684@qq.com>
// Licensed under Apache-2.0, see LICENSE file for full license terms.

use std::fmt;

/// Central error type for the anthropic-rs client.
#[derive(Debug)]
pub enum AnthropicError {
    /// Configuration problem (missing api key, invalid base_url, etc.)
    Config(String),
    /// Network / HTTP transport failure (DNS, connection, timeout).
    Network(String),
    /// API returned an error status (4xx, 5xx) or a streamed error event.
    Api(String),
    /// JSON (de)serialization failure.
    Json(String),
    /// I/O failure (reading response body, writing to sink).
    Io(String),
    /// The operation was cancelled by the caller.
    Cancelled,
}

impl fmt::Display for AnthropicError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AnthropicError::Config(m) => write!(f, "config error: {m}"),
            AnthropicError::Network(m) => write!(f, "network error: {m}"),
            AnthropicError::Api(m) => write!(f, "api error: {m}"),
            AnthropicError::Json(m) => write!(f, "json error: {m}"),
            AnthropicError::Io(m) => write!(f, "io error: {m}"),
            AnthropicError::Cancelled => write!(f, "cancelled"),
        }
    }
}

impl std::error::Error for AnthropicError {}

impl From<serde_json::Error> for AnthropicError {
    fn from(e: serde_json::Error) -> Self {
        AnthropicError::Json(e.to_string())
    }
}

impl From<std::io::Error> for AnthropicError {
    fn from(e: std::io::Error) -> Self {
        AnthropicError::Io(e.to_string())
    }
}

#[cfg(feature = "async")]
impl From<reqwest::Error> for AnthropicError {
    fn from(e: reqwest::Error) -> Self {
        AnthropicError::Network(e.to_string())
    }
}

/// Convenience result alias.
pub type Result<T> = std::result::Result<T, AnthropicError>;

impl AnthropicError {
    /// Whether this error is worth retrying.
    pub fn is_retryable(&self) -> bool {
        match self {
            AnthropicError::Network(_) => true,
            AnthropicError::Json(_) => true,
            AnthropicError::Api(m) => {
                let l = m.to_lowercase();
                l.contains("timeout")
                    || l.contains("timed out")
                    || l.contains("connection")
                    || l.contains(" 500")
                    || l.contains(" 502")
                    || l.contains(" 503")
                    || l.contains(" 504")
                    || l.contains("rate limit")
            }
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_formats() {
        assert_eq!(AnthropicError::Cancelled.to_string(), "cancelled");
        assert!(AnthropicError::Config("x".into()).to_string().contains("config"));
    }

    #[test]
    fn retryable_classification() {
        assert!(AnthropicError::Network("reset".into()).is_retryable());
        assert!(AnthropicError::Api("HTTP 503 unavailable".into()).is_retryable());
        assert!(AnthropicError::Api("rate limit exceeded".into()).is_retryable());
        assert!(AnthropicError::Json("malformed response".into()).is_retryable());
        assert!(!AnthropicError::Api("HTTP 401 unauthorized".into()).is_retryable());
        assert!(!AnthropicError::Config("no key".into()).is_retryable());
    }

    #[test]
    fn json_error_converts() {
        let e: AnthropicError = serde_json::from_str::<i32>("not json").unwrap_err().into();
        matches!(e, AnthropicError::Json(_));
    }
}