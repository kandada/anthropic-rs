// Copyright (c) 2025 xiefujin <490021684@qq.com>
// Licensed under Apache-2.0, see LICENSE file for full license terms.

use std::time::Duration;

use crate::error::{AnthropicError, Result};
use crate::retry::RetryConfig;

/// Connect timeout: fail fast on dead links.
const CONNECT_TIMEOUT_SECS: u64 = 10;
/// Per-read socket timeout for the SSE stream.
const DEFAULT_READ_TIMEOUT_SECS: u64 = 30;
const WRITE_TIMEOUT_SECS: u64 = 30;

fn build_agent(read_timeout_secs: u64, total_timeout_secs: u64, proxy: Option<&str>) -> ureq::Agent {
    let mut b = ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(CONNECT_TIMEOUT_SECS))
        .timeout_read(Duration::from_secs(read_timeout_secs))
        .timeout_write(Duration::from_secs(WRITE_TIMEOUT_SECS))
        .timeout(Duration::from_secs(total_timeout_secs));
    if let Some(p) = proxy {
        b = b.proxy(ureq::Proxy::new(p).expect("invalid proxy URL"));
    }
    b.build()
}

/// An Anthropic-compatible API client.
///
/// Works with Anthropic Claude, MiniMax, DeepSeek-anthropic, and
/// any provider implementing the Anthropic Messages API.
pub struct AnthropicClient {
    api_key: String,
    model: String,
    base_url: String,
    agent: ureq::Agent,
    /// Anthropic API version header.
    api_version: String,
    /// Default max_tokens.
    default_max_tokens: u64,
    /// Automatic retry configuration (on by default).
    retry_config: RetryConfig,
    /// HTTP proxy URL (e.g. `http://localhost:8080`).
    proxy: Option<String>,
}

impl AnthropicClient {
    /// Create a client for Anthropic's default API.
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        let api_key = api_key.into();
        let base_url = Self::detect_base_url(&api_key);
        AnthropicClient {
            api_key,
            model: model.into(),
            base_url,
            agent: build_agent(DEFAULT_READ_TIMEOUT_SECS, 300, None),
            api_version: "2023-06-01".into(),
            default_max_tokens: 4096,
            retry_config: RetryConfig::default(),
            proxy: None,
        }
    }

    /// Create a client with a custom base URL for non-Anthropic providers.
    pub fn with_base_url(
        api_key: impl Into<String>,
        model: impl Into<String>,
        base_url: impl Into<String>,
    ) -> Self {
        AnthropicClient {
            api_key: api_key.into(),
            model: model.into(),
            base_url: base_url.into(),
            agent: build_agent(DEFAULT_READ_TIMEOUT_SECS, 300, None),
            api_version: "2023-06-01".into(),
            default_max_tokens: 4096,
            retry_config: RetryConfig::default(),
            proxy: None,
        }
    }

    /// Route requests through an HTTP proxy.
    pub fn with_proxy(mut self, proxy: impl Into<String>) -> Self {
        self.proxy = Some(proxy.into());
        self.agent = build_agent(DEFAULT_READ_TIMEOUT_SECS, 300, self.proxy.as_deref());
        self
    }

    /// Set the Anthropic API version.
    pub fn with_api_version(mut self, version: impl Into<String>) -> Self {
        self.api_version = version.into();
        self
    }

    /// Set the default max_tokens.
    pub fn with_max_tokens(mut self, max_tokens: u64) -> Self {
        self.default_max_tokens = max_tokens;
        self
    }

    /// Set the maximum number of automatic retries (0 disables retry).
    pub fn with_retries(mut self, max_retries: u32) -> Self {
        self.retry_config.max_retries = max_retries;
        self
    }

    /// Set a full retry configuration.
    pub fn with_retry_config(mut self, config: RetryConfig) -> Self {
        self.retry_config = config;
        self
    }

    /// Override the read timeout (seconds).
    pub fn with_read_timeout(mut self, secs: u64) -> Self {
        let total = 300;
        self.agent = build_agent(secs, total, self.proxy.as_deref());
        self
    }

    /// Override the total request timeout (seconds).
    pub fn with_total_timeout(mut self, secs: u64) -> Self {
        let read = DEFAULT_READ_TIMEOUT_SECS;
        self.agent = build_agent(read, secs, self.proxy.as_deref());
        self
    }

    /// Change the model after construction.
    pub fn set_model(&mut self, model: impl Into<String>) {
        self.model = model.into();
    }

    /// The current model name.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Detect the default base URL based on API key prefix.
    fn detect_base_url(api_key: &str) -> String {
        if api_key.starts_with("sk-") && !api_key.starts_with("sk-ant") {
            "https://api.deepseek.com/anthropic".into()
        } else {
            "https://api.anthropic.com".into()
        }
    }

    pub fn endpoint(&self) -> String {
        format!("{}/v1/messages", self.base_url.trim_end_matches('/'))
    }

    pub(crate) fn api_key(&self) -> &str {
        &self.api_key
    }

    pub(crate) fn api_version(&self) -> &str {
        &self.api_version
    }

    /// Validate the API key with a minimal request.
    pub fn validate(&self) -> Result<()> {
        let body = serde_json::json!({
            "model": self.model,
            "max_tokens": 4,
            "messages": [{"role": "user", "content": "Hi"}],
        });
        self.post_json(body).map(|_| ())
    }

    pub fn post_json(&self, body: serde_json::Value) -> Result<ureq::Response> {
        retry_response(&self.retry_config, || self.post_json_once(&body))
    }

    fn post_json_once(&self, body: &serde_json::Value) -> Result<ureq::Response> {
        match self
            .agent
            .post(&self.endpoint())
            .set("x-api-key", self.api_key())
            .set("anthropic-version", self.api_version())
            .set("Content-Type", "application/json")
            .send_json(body.clone())
        {
            Ok(r) => Ok(r),
            Err(ureq::Error::Status(code, r)) => {
                let retry_after = parse_retry_after(&r);
                let msg = r.into_string().unwrap_or_default();
                Err(AnthropicError::api(code, retry_after, truncate(&msg, 500)))
            }
            Err(e) => Err(AnthropicError::Network(e.to_string())),
        }
    }

    /// POST to an arbitrary path under the base URL (e.g.
    /// `v1/messages/count_tokens`), with the same retry/error handling.
    pub(crate) fn post_json_path(&self, path: &str, body: serde_json::Value) -> Result<ureq::Response> {
        let url = format!("{}/{}", self.base_url.trim_end_matches('/'), path.trim_start_matches('/'));
        retry_response(&self.retry_config, || {
            match self
                .agent
                .post(&url)
                .set("x-api-key", self.api_key())
                .set("anthropic-version", self.api_version())
                .set("Content-Type", "application/json")
                .send_json(body.clone())
            {
                Ok(r) => Ok(r),
                Err(ureq::Error::Status(code, r)) => {
                    let retry_after = parse_retry_after(&r);
                    let msg = r.into_string().unwrap_or_default();
                    Err(AnthropicError::api(code, retry_after, truncate(&msg, 500)))
                }
                Err(e) => Err(AnthropicError::Network(e.to_string())),
            }
        })
    }

    pub(crate) fn post_stream(&self, body: serde_json::Value) -> Result<impl std::io::Read> {
        retry_response(&self.retry_config, || self.post_stream_once(&body)).map(|r| r.into_reader())
    }

    fn post_stream_once(&self, body: &serde_json::Value) -> Result<ureq::Response> {
        match self
            .agent
            .post(&self.endpoint())
            .set("x-api-key", self.api_key())
            .set("anthropic-version", self.api_version())
            .set("Content-Type", "application/json")
            .set("Accept", "text/event-stream")
            .send_json(body.clone())
        {
            Ok(r) => Ok(r),
            Err(ureq::Error::Status(code, r)) => {
                let retry_after = parse_retry_after(&r);
                let msg = r.into_string().unwrap_or_default();
                Err(AnthropicError::api(code, retry_after, truncate(&msg, 500)))
            }
            Err(e) => Err(AnthropicError::Network(e.to_string())),
        }
    }
}

/// Run a single-shot HTTP closure with the retry policy.
fn retry_response<F>(config: &RetryConfig, mut once: F) -> Result<ureq::Response>
where
    F: FnMut() -> Result<ureq::Response>,
{
    let mut attempt = 0u32;
    loop {
        match once() {
            Ok(r) => return Ok(r),
            Err(e) => {
                if attempt >= config.max_retries || !e.is_retryable() {
                    return Err(e);
                }
                let delay = match e.retry_after_secs() {
                    Some(secs) => Duration::from_secs(secs),
                    None => Duration::from_millis(config.delay_ms(attempt)),
                };
                attempt += 1;
                std::thread::sleep(delay);
            }
        }
    }
}

fn parse_retry_after(r: &ureq::Response) -> Option<u64> {
    r.header("Retry-After")
        .and_then(|v| v.trim().parse::<u64>().ok())
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        let head: String = s.chars().take(n).collect();
        format!("{head}...")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn client_new() {
        let client = AnthropicClient::new("sk-ant-test", "claude-sonnet-4-20250514");
        assert_eq!(client.model(), "claude-sonnet-4-20250514");
    }

    #[test]
    fn client_with_base_url() {
        let client = AnthropicClient::with_base_url(
            "sk-test",
            "deepseek-chat",
            "https://api.deepseek.com/anthropic",
        );
        assert_eq!(
            client.endpoint(),
            "https://api.deepseek.com/anthropic/v1/messages"
        );
    }

    #[test]
    fn client_custom_timeout() {
        let _client = AnthropicClient::new("sk-ant-test", "test")
            .with_read_timeout(60)
            .with_total_timeout(600);
    }

    #[test]
    fn client_retry_config_default_on() {
        let client = AnthropicClient::new("sk-ant-test", "test");
        assert_eq!(client.retry_config.max_retries, 3);
        let disabled = client.with_retries(0);
        assert_eq!(disabled.retry_config.max_retries, 0);
    }

    #[test]
    fn client_proxy_accepted() {
        let client = AnthropicClient::new("sk-ant-test", "test")
            .with_proxy("http://127.0.0.1:8080");
        assert_eq!(client.proxy.as_deref(), Some("http://127.0.0.1:8080"));
    }
}