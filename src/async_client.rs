// Copyright (c) 2025 xiefujin <490021684@qq.com>
// Licensed under Apache-2.0, see LICENSE file for full license terms.

//! Async Anthropic client using `reqwest`.

use std::time::Duration;

use crate::error::{AnthropicError, Result};
use crate::retry::RetryConfig;

fn build_http_client(timeout_secs: u64, proxy: Option<&str>) -> reqwest::Client {
    let mut b = reqwest::Client::builder()
        .timeout(Duration::from_secs(timeout_secs))
        .connect_timeout(Duration::from_secs(10));
    if let Some(p) = proxy {
        b = b.proxy(reqwest::Proxy::all(p).expect("invalid proxy URL"));
    }
    b.build().expect("failed to build reqwest client")
}

/// An async Anthropic-compatible API client using `reqwest`.
pub struct AnthropicAsyncClient {
    api_key: String,
    model: String,
    base_url: String,
    client: reqwest::Client,
    api_version: String,
    default_max_tokens: u64,
    retry_config: RetryConfig,
    proxy: Option<String>,
}

impl AnthropicAsyncClient {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        let api_key = api_key.into();
        let base_url = Self::detect_base_url(&api_key);
        AnthropicAsyncClient {
            api_key,
            model: model.into(),
            base_url,
            client: build_http_client(300, None),
            api_version: "2023-06-01".into(),
            default_max_tokens: 4096,
            retry_config: RetryConfig::default(),
            proxy: None,
        }
    }

    pub fn with_base_url(api_key: impl Into<String>, model: impl Into<String>, base_url: impl Into<String>) -> Self {
        AnthropicAsyncClient {
            api_key: api_key.into(),
            model: model.into(),
            base_url: base_url.into(),
            client: build_http_client(300, None),
            api_version: "2023-06-01".into(),
            default_max_tokens: 4096,
            retry_config: RetryConfig::default(),
            proxy: None,
        }
    }

    pub fn with_max_tokens(mut self, max_tokens: u64) -> Self { self.default_max_tokens = max_tokens; self }
    pub fn with_api_version(mut self, version: impl Into<String>) -> Self { self.api_version = version.into(); self }
    pub fn with_timeout(mut self, secs: u64) -> Self {
        self.client = build_http_client(secs, self.proxy.as_deref());
        self
    }
    /// Route requests through an HTTP proxy.
    pub fn with_proxy(mut self, proxy: impl Into<String>) -> Self {
        self.proxy = Some(proxy.into());
        self.client = build_http_client(300, self.proxy.as_deref());
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
    pub fn model(&self) -> &str { &self.model }
    pub fn set_model(&mut self, model: impl Into<String>) { self.model = model.into(); }

    pub(crate) fn endpoint(&self) -> String {
        format!("{}/v1/messages", self.base_url.trim_end_matches('/'))
    }

    fn detect_base_url(api_key: &str) -> String {
        if api_key.starts_with("sk-") && !api_key.starts_with("sk-ant") {
            "https://api.deepseek.com/anthropic".into()
        } else { "https://api.anthropic.com".into() }
    }

    pub async fn validate(&self) -> Result<()> {
        let body = serde_json::json!({
            "model": self.model, "max_tokens": 4,
            "messages": [{"role": "user", "content": "Hi"}],
        });
        self.post_json(body).await.map(|_| ())
    }

    pub(crate) async fn post_json(&self, body: serde_json::Value) -> Result<reqwest::Response> {
        let url = self.endpoint();
        retry_async_response(&self.retry_config, || {
            self.send_json(&url, &body, false)
        }).await
    }

    /// POST to an arbitrary path under the base URL (e.g.
    /// `v1/messages/count_tokens`), with the same retry/error handling.
    pub(crate) async fn post_json_path(&self, path: &str, body: serde_json::Value) -> Result<reqwest::Response> {
        let url = format!("{}/{}", self.base_url.trim_end_matches('/'), path.trim_start_matches('/'));
        retry_async_response(&self.retry_config, || {
            self.send_json(&url, &body, false)
        }).await
    }

    pub(crate) async fn post_stream(&self, body: serde_json::Value) -> Result<reqwest::Response> {
        let url = self.endpoint();
        retry_async_response(&self.retry_config, || {
            self.send_json(&url, &body, true)
        }).await
    }

    async fn send_json(
        &self,
        url: &str,
        body: &serde_json::Value,
        stream: bool,
    ) -> Result<reqwest::Response> {
        let mut req = self
            .client
            .post(url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", &self.api_version)
            .header("Content-Type", "application/json")
            .json(body);
        if stream {
            req = req.header("Accept", "text/event-stream");
        }
        let resp = req.send().await.map_err(|e| AnthropicError::Network(e.to_string()))?;
        if !resp.status().is_success() {
            let code = resp.status().as_u16();
            let retry_after = resp
                .headers()
                .get(reqwest::header::RETRY_AFTER)
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.trim().parse::<u64>().ok());
            let msg = resp.text().await.unwrap_or_default();
            return Err(AnthropicError::api(code, retry_after, truncate(&msg, 500)));
        }
        Ok(resp)
    }
}

/// Run a single-shot async HTTP closure with the retry policy.
async fn retry_async_response<F, Fut>(config: &RetryConfig, mut once: F) -> Result<reqwest::Response>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<reqwest::Response>>,
{
    let mut attempt = 0u32;
    loop {
        match once().await {
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
                tokio::time::sleep(delay).await;
            }
        }
    }
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        let head: String = s.chars().take(n).collect();
        format!("{head}...")
    }
}
