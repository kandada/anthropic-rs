// Copyright (c) 2025 xiefujin <490021684@qq.com>
// Licensed under Apache-2.0, see LICENSE file for full license terms.

//! Async messages API for Anthropic.

use std::collections::BTreeMap;
use serde_json::Value;

use crate::api_common::{build_message_body, assemble_message_response};
use crate::async_client::AnthropicAsyncClient;
use crate::async_sse::AsyncSseStream;
use crate::error::{AnthropicError, Result};
use crate::types::{ChatMessage, LlmResponse, SimplifiedToolCall, Tool};

#[derive(Default, Clone)]
struct BlockAcc {
    kind: String,
    id: String,
    name: String,
    text: String,
    partial_json: String,
}

impl AnthropicAsyncClient {
    /// Create a message (non-streaming).
    pub async fn messages_create(
        &self,
        messages: &[ChatMessage],
        system: Option<&str>,
        tools: Option<&[Tool]>,
        max_tokens: u64,
    ) -> Result<LlmResponse> {
        let body = build_message_body(self.model(), messages, system, tools, max_tokens, false);
        let resp = self.post_json(body).await?;
        let raw: Value = resp.json().await?;
        Ok(assemble_message_response(&raw))
    }

    /// Stream a message.
    pub async fn messages_stream(
        &self,
        messages: &[ChatMessage],
        system: Option<&str>,
        tools: Option<&[Tool]>,
        max_tokens: u64,
        mut on_delta: impl FnMut(&str),
        mut on_tool_call: impl FnMut(&str, &str),
    ) -> Result<LlmResponse> {
        let body = build_message_body(self.model(), messages, system, tools, max_tokens, true);
        let resp = self.post_stream(body).await?;
        let stream = resp.bytes_stream();
        let mut sse = AsyncSseStream::new(stream);

        let mut blocks: BTreeMap<i64, BlockAcc> = BTreeMap::new();
        let mut text = String::new();
        let mut reasoning = String::new();
        let mut stop_reason: Option<String> = None;
        let mut valid_chunks: usize = 0;
        let mut total_payloads: usize = 0;

        while let Some(payload) = sse.next_data().await? {
            total_payloads += 1;
            let ev: Value = match serde_json::from_str(&payload) {
                Ok(v) => { valid_chunks += 1; v }
                Err(_) => continue,
            };
            let etype = ev.get("type").and_then(|v| v.as_str()).unwrap_or("");
            match etype {
                "error" => {
                    let msg = ev.get("error").and_then(|e| e.get("message"))
                        .and_then(|v| v.as_str()).map(|s| s.to_string())
                        .unwrap_or_else(|| ev.to_string());
                    return Err(AnthropicError::Api(format!("stream error: {msg}")));
                }
                "content_block_start" => {
                    let idx = ev.get("index").and_then(|v| v.as_i64()).unwrap_or(0);
                    let block = ev.get("content_block");
                    let mut acc = BlockAcc::default();
                    if let Some(b) = block {
                        acc.kind = b.get("type").and_then(|v| v.as_str()).unwrap_or("").to_string();
                        acc.id = b.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
                        acc.name = b.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string();
                    }
                    blocks.insert(idx, acc);
                }
                "content_block_delta" => {
                    let idx = ev.get("index").and_then(|v| v.as_i64()).unwrap_or(0);
                    let delta = ev.get("delta");
                    if let Some(d) = delta {
                        let dt = d.get("type").and_then(|v| v.as_str()).unwrap_or("");
                        let acc = blocks.entry(idx).or_default();
                        match dt {
                            "text_delta" => {
                                if let Some(t) = d.get("text").and_then(|v| v.as_str()) {
                                    text.push_str(t); acc.text.push_str(t); on_delta(t);
                                }
                            }
                            "thinking_delta" => {
                                if let Some(t) = d.get("thinking").and_then(|v| v.as_str()) {
                                    reasoning.push_str(t); acc.text.push_str(t); on_delta(t);
                                }
                            }
                            "input_json_delta" => {
                                if let Some(pj) = d.get("partial_json").and_then(|v| v.as_str()) {
                                    acc.partial_json.push_str(pj);
                                }
                            }
                            _ => {}
                        }
                    }
                }
                "message_delta" => {
                    if let Some(sr) = ev.get("delta").and_then(|d| d.get("stop_reason")).and_then(|v| v.as_str()) {
                        stop_reason = Some(sr.to_string());
                    }
                }
                _ => {}
            }
        }

        if total_payloads > 0 && valid_chunks == 0 {
            return Err(AnthropicError::Api("stream returned no parseable data".into()));
        }
        if stop_reason.is_none() && (!text.is_empty() || !reasoning.is_empty() || !blocks.is_empty()) {
            stop_reason = Some("connection_closed".to_string());
        }

        let mut tool_calls = Vec::new();
        for (i, (_, acc)) in blocks.into_iter().enumerate() {
            if acc.kind != "tool_use" || acc.name.is_empty() { continue; }
            let args = if acc.partial_json.trim().is_empty() { "{}".to_string() } else { acc.partial_json };
            let id = if acc.id.is_empty() { format!("call_{i}") } else { acc.id };
            on_tool_call(&acc.name, &args);
            tool_calls.push(SimplifiedToolCall { id, name: acc.name, arguments: args });
        }

        if matches!(stop_reason.as_deref(), Some("max_tokens")) {
            text.push_str("\n\n[WARNING: API response truncated (max_tokens).]");
        }

        Ok(LlmResponse {
            text, tool_calls,
            reasoning_content: if reasoning.is_empty() { None } else { Some(reasoning) },
            finish_reason: stop_reason, usage: None,
        })
    }
}