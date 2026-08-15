// Copyright (c) 2025 xiefujin <490021684@qq.com>
// Licensed under Apache-2.0, see LICENSE file for full license terms.

//! Async Anthropic client using `reqwest`.

use std::time::Duration;

use crate::error::{AnthropicError, Result};

/// An async Anthropic-compatible API client using `reqwest`.
pub struct AnthropicAsyncClient {
    api_key: String,
    model: String,
    base_url: String,
    client: reqwest::Client,
    api_version: String,
    default_max_tokens: u64,
}

impl AnthropicAsyncClient {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        let api_key = api_key.into();
        let base_url = Self::detect_base_url(&api_key);
        AnthropicAsyncClient {
            api_key,
            model: model.into(),
            base_url,
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(300))
                .connect_timeout(Duration::from_secs(10))
                .build().expect("failed to build reqwest client"),
            api_version: "2023-06-01".into(),
            default_max_tokens: 4096,
        }
    }

    pub fn with_base_url(api_key: impl Into<String>, model: impl Into<String>, base_url: impl Into<String>) -> Self {
        AnthropicAsyncClient {
            api_key: api_key.into(),
            model: model.into(),
            base_url: base_url.into(),
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(300))
                .connect_timeout(Duration::from_secs(10))
                .build().expect("failed to build reqwest client"),
            api_version: "2023-06-01".into(),
            default_max_tokens: 4096,
        }
    }

    pub fn with_max_tokens(mut self, max_tokens: u64) -> Self { self.default_max_tokens = max_tokens; self }
    pub fn with_api_version(mut self, version: impl Into<String>) -> Self { self.api_version = version.into(); self }
    pub fn with_timeout(mut self, secs: u64) -> Self {
        self.client = reqwest::Client::builder().timeout(Duration::from_secs(secs)).connect_timeout(Duration::from_secs(10)).build().expect("failed to build reqwest client");
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
        let resp = self.client.post(self.endpoint())
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", &self.api_version)
            .header("Content-Type", "application/json")
            .json(&body).send().await
            .map_err(|e| AnthropicError::Network(e.to_string()))?;
        if !resp.status().is_success() {
            let code = resp.status().as_u16();
            let msg = resp.text().await.unwrap_or_default();
            return Err(AnthropicError::Api(format!("HTTP {code}: {}", truncate(&msg, 500))));
        }
        Ok(resp)
    }

    pub(crate) async fn post_stream(&self, body: serde_json::Value) -> Result<reqwest::Response> {
        let resp = self.client.post(self.endpoint())
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", &self.api_version)
            .header("Content-Type", "application/json")
            .header("Accept", "text/event-stream")
            .json(&body).send().await
            .map_err(|e| AnthropicError::Network(e.to_string()))?;
        if !resp.status().is_success() {
            let code = resp.status().as_u16();
            let msg = resp.text().await.unwrap_or_default();
            return Err(AnthropicError::Api(format!("HTTP {code}: {}", truncate(&msg, 500))));
        }
        Ok(resp)
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
