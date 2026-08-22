// Copyright (c) 2025 xiefujin <490021684@qq.com>
// Licensed under Apache-2.0, see LICENSE file for full license terms.

//! Async messages API for Anthropic.

use std::collections::BTreeMap;
use std::future::Future;
use serde_json::Value;

use crate::api_common::{build_message_body, assemble_message_response};
use crate::async_client::AnthropicAsyncClient;
use crate::async_sse::AsyncSseStream;
use crate::error::{AnthropicError, Result};
use crate::request::{MessageRequest, ThinkingConfig};
use crate::types::{ChatMessage, ContentBlock, CountTokensResponse, LlmResponse, SimplifiedToolCall, StreamEvent, Tool, Usage};

/// Async stream of raw, typed [`StreamEvent`]s.
pub struct AsyncStreamEventStream<S> {
    sse: AsyncSseStream<S>,
}

impl<S> AsyncStreamEventStream<S>
where
    S: futures::Stream<Item = std::result::Result<bytes::Bytes, reqwest::Error>> + Unpin,
{
    pub fn new(stream: S) -> Self {
        AsyncStreamEventStream { sse: AsyncSseStream::new(stream) }
    }
}

impl<S> futures::Stream for AsyncStreamEventStream<S>
where
    S: futures::Stream<Item = std::result::Result<bytes::Bytes, reqwest::Error>> + Unpin,
{
    type Item = std::result::Result<StreamEvent, AnthropicError>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        let fut = self.sse.next_data();
        futures::pin_mut!(fut);
        match fut.poll(cx) {
            std::task::Poll::Ready(Ok(Some(payload))) => {
                let item = match serde_json::from_str(&payload) {
                    Ok(ev) => Ok(ev),
                    Err(e) => Err(AnthropicError::Json(e.to_string())),
                };
                std::task::Poll::Ready(Some(item))
            }
            std::task::Poll::Ready(Ok(None)) => std::task::Poll::Ready(None),
            std::task::Poll::Ready(Err(e)) => std::task::Poll::Ready(Some(Err(e))),
            std::task::Poll::Pending => std::task::Poll::Pending,
        }
    }
}

#[derive(Default, Clone)]
struct BlockAcc {
    kind: String,
    id: String,
    name: String,
    text: String,
    thinking: String,
    signature: String,
    data: String,
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
        thinking: Option<&ThinkingConfig>,
    ) -> Result<LlmResponse> {
        let body = build_message_body(self.model(), messages, system, tools, max_tokens, false, thinking);
        let resp = self.post_json(body).await?;
        let raw: Value = resp.json().await?;
        Ok(assemble_message_response(&raw))
    }

    /// Count input tokens via the official `messages/count_tokens` endpoint.
    pub async fn count_tokens(
        &self,
        messages: &[ChatMessage],
        system: Option<&str>,
        tools: Option<&[Tool]>,
    ) -> Result<CountTokensResponse> {
        let body = crate::api_common::build_count_tokens_body(self.model(), messages, system, tools);
        let resp = self.post_json_path("v1/messages/count_tokens", body).await?;
        Ok(resp.json().await?)
    }

    /// Stream and return raw typed [`StreamEvent`]s.
    pub async fn messages_stream_events(
        &self,
        messages: &[ChatMessage],
        system: Option<&str>,
        tools: Option<&[Tool]>,
        max_tokens: u64,
        thinking: Option<&ThinkingConfig>,
    ) -> Result<AsyncStreamEventStream<
        std::pin::Pin<
            Box<dyn futures::Stream<Item = std::result::Result<bytes::Bytes, reqwest::Error>> + Send>,
        >,
    >> {
        let body = build_message_body(self.model(), messages, system, tools, max_tokens, true, thinking);
        let resp = self.post_stream(body).await?;
        let s: std::pin::Pin<
            Box<dyn futures::Stream<Item = std::result::Result<bytes::Bytes, reqwest::Error>> + Send>,
        > = Box::pin(resp.bytes_stream());
        Ok(AsyncStreamEventStream::new(s))
    }

    /// Stream a message.
    #[allow(clippy::too_many_arguments)]
    pub async fn messages_stream(
        &self,
        messages: &[ChatMessage],
        system: Option<&str>,
        tools: Option<&[Tool]>,
        max_tokens: u64,
        thinking: Option<&ThinkingConfig>,
        mut on_delta: impl FnMut(&str),
        mut on_tool_call: impl FnMut(&str, &str),
    ) -> Result<LlmResponse> {
        let body = build_message_body(self.model(), messages, system, tools, max_tokens, true, thinking);
        let resp = self.post_stream(body).await?;
        let stream = resp.bytes_stream();
        let mut sse = AsyncSseStream::new(stream);

        let mut blocks: BTreeMap<i64, BlockAcc> = BTreeMap::new();
        let mut text = String::new();
        let mut reasoning = String::new();
        let mut stop_reason: Option<String> = None;
        let mut usage: Option<Usage> = None;
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
                    return Err(AnthropicError::stream_error(format!("stream error: {msg}")));
                }
                "message_start" => {
                    if let Some(u) = ev
                        .get("message")
                        .and_then(|m| m.get("usage"))
                        .and_then(|u| serde_json::from_value(u.clone()).ok())
                    {
                        usage = Some(u);
                    }
                }
                "content_block_start" => {
                    let idx = ev.get("index").and_then(|v| v.as_i64()).unwrap_or(0);
                    let block = ev.get("content_block");
                    let mut acc = BlockAcc::default();
                    if let Some(b) = block {
                        acc.kind = b.get("type").and_then(|v| v.as_str()).unwrap_or("").to_string();
                        acc.id = b.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
                        acc.name = b.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string();
                        acc.data = b.get("data").and_then(|v| v.as_str()).unwrap_or("").to_string();
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
                                    reasoning.push_str(t); acc.thinking.push_str(t); on_delta(t);
                                }
                            }
                            "signature_delta" => {
                                if let Some(sig) = d.get("signature").and_then(|v| v.as_str()) {
                                    acc.signature = sig.to_string();
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
                    if let Some(u) = ev
                        .get("usage")
                        .and_then(|u| serde_json::from_value(u.clone()).ok())
                    {
                        usage = Some(u);
                    }
                }
                _ => {}
            }
        }

        if total_payloads > 0 && valid_chunks == 0 {
            return Err(AnthropicError::stream_error("stream returned no parseable data"));
        }
        if stop_reason.is_none() && (!text.is_empty() || !reasoning.is_empty() || !blocks.is_empty()) {
            stop_reason = Some("connection_closed".to_string());
        }

        let mut tool_calls = Vec::new();
        let mut content_blocks: Vec<ContentBlock> = Vec::new();
        for (i, (_, acc)) in blocks.into_iter().enumerate() {
            match acc.kind.as_str() {
                "text" => {
                    if !acc.text.is_empty() {
                        content_blocks.push(ContentBlock::Text { text: acc.text, citations: None });
                    }
                }
                "thinking" => {
                    if !acc.thinking.is_empty() {
                        content_blocks.push(ContentBlock::Thinking {
                            thinking: acc.thinking,
                            signature: acc.signature,
                        });
                    }
                }
                "redacted_thinking" => {
                    content_blocks.push(ContentBlock::RedactedThinking { data: acc.data });
                }
                "tool_use" => {
                    if acc.name.is_empty() { continue; }
                    let args = if acc.partial_json.trim().is_empty() { "{}".to_string() } else { acc.partial_json };
                    let id = if acc.id.is_empty() { format!("call_{i}") } else { acc.id };
                    on_tool_call(&acc.name, &args);
                    tool_calls.push(SimplifiedToolCall {
                        id: id.clone(),
                        name: acc.name.clone(),
                        arguments: args.clone(),
                    });
                    content_blocks.push(ContentBlock::ToolUse {
                        id,
                        name: acc.name,
                        input: serde_json::from_str(&args).unwrap_or_else(|_| serde_json::json!({})),
                    });
                }
                _ => {}
            }
        }

        if matches!(stop_reason.as_deref(), Some("max_tokens")) {
            text.push_str("\n\n[WARNING: API response truncated (max_tokens).]");
        }

        Ok(LlmResponse {
            text, tool_calls,
            reasoning_content: if reasoning.is_empty() { None } else { Some(reasoning) },
            finish_reason: stop_reason,
            usage,
            content_blocks,
        })
    }

    /// Send a [`MessageRequest`] (non-streaming).
    pub async fn messages_send(&self, request: &MessageRequest) -> Result<LlmResponse> {
        let mut body = request.build_body();
        body["stream"] = serde_json::json!(false);
        let resp = self.post_json(body).await?;
        let raw: Value = resp.json().await?;
        Ok(assemble_message_response(&raw))
    }

    /// Send a [`MessageRequest`] with streaming.
    pub async fn messages_send_stream(
        &self,
        request: &MessageRequest,
        mut on_delta: impl FnMut(&str),
        mut on_tool_call: impl FnMut(&str, &str),
    ) -> Result<LlmResponse> {
        let mut body = request.build_body();
        body["stream"] = serde_json::json!(true);
        let resp = self.post_stream(body).await?;
        let stream = resp.bytes_stream();
        let mut sse = AsyncSseStream::new(stream);
        parse_async_stream(&mut sse, &mut on_delta, &mut on_tool_call).await
    }
}

async fn parse_async_stream<S>(
    sse: &mut AsyncSseStream<S>,
    on_delta: &mut impl FnMut(&str),
    on_tool_call: &mut impl FnMut(&str, &str),
) -> Result<LlmResponse>
where
    S: futures::Stream<Item = std::result::Result<bytes::Bytes, reqwest::Error>> + Unpin,
{
    let mut blocks: BTreeMap<i64, BlockAcc> = BTreeMap::new();
    let mut text = String::new();
    let mut reasoning = String::new();
    let mut stop_reason: Option<String> = None;
    let mut usage: Option<Usage> = None;
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
                return Err(AnthropicError::stream_error(format!("stream error: {msg}")));
            }
            "message_start" => {
                if let Some(u) = ev
                    .get("message")
                    .and_then(|m| m.get("usage"))
                    .and_then(|u| serde_json::from_value(u.clone()).ok())
                {
                    usage = Some(u);
                }
            }
            "content_block_start" => {
                let idx = ev.get("index").and_then(|v| v.as_i64()).unwrap_or(0);
                let block = ev.get("content_block");
                let mut acc = BlockAcc::default();
                if let Some(b) = block {
                    acc.kind = b.get("type").and_then(|v| v.as_str()).unwrap_or("").to_string();
                    acc.id = b.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
                    acc.name = b.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string();
                    acc.data = b.get("data").and_then(|v| v.as_str()).unwrap_or("").to_string();
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
                                reasoning.push_str(t); acc.thinking.push_str(t); on_delta(t);
                            }
                        }
                        "signature_delta" => {
                            if let Some(sig) = d.get("signature").and_then(|v| v.as_str()) {
                                acc.signature = sig.to_string();
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
                if let Some(u) = ev
                    .get("usage")
                    .and_then(|u| serde_json::from_value(u.clone()).ok())
                {
                    usage = Some(u);
                }
            }
            _ => {}
        }
    }

    if total_payloads > 0 && valid_chunks == 0 {
        return Err(AnthropicError::stream_error("stream returned no parseable data"));
    }
    if stop_reason.is_none() && (!text.is_empty() || !reasoning.is_empty() || !blocks.is_empty()) {
        stop_reason = Some("connection_closed".to_string());
    }

    let mut tool_calls = Vec::new();
    let mut content_blocks: Vec<ContentBlock> = Vec::new();
    for (i, (_, acc)) in blocks.into_iter().enumerate() {
        match acc.kind.as_str() {
            "text" => {
                if !acc.text.is_empty() {
                    content_blocks.push(ContentBlock::Text { text: acc.text, citations: None });
                }
            }
            "thinking" => {
                if !acc.thinking.is_empty() {
                    content_blocks.push(ContentBlock::Thinking {
                        thinking: acc.thinking,
                        signature: acc.signature,
                    });
                }
            }
            "redacted_thinking" => {
                content_blocks.push(ContentBlock::RedactedThinking { data: acc.data });
            }
            "tool_use" => {
                if acc.name.is_empty() { continue; }
                let args = if acc.partial_json.trim().is_empty() { "{}".to_string() } else { acc.partial_json };
                let id = if acc.id.is_empty() { format!("call_{i}") } else { acc.id };
                on_tool_call(&acc.name, &args);
                tool_calls.push(SimplifiedToolCall {
                    id: id.clone(),
                    name: acc.name.clone(),
                    arguments: args.clone(),
                });
                content_blocks.push(ContentBlock::ToolUse {
                    id,
                    name: acc.name,
                    input: serde_json::from_str(&args).unwrap_or_else(|_| serde_json::json!({})),
                });
            }
            _ => {}
        }
    }

    if matches!(stop_reason.as_deref(), Some("max_tokens")) {
        text.push_str("\n\n[WARNING: API response truncated (max_tokens).]");
    }

    Ok(LlmResponse {
        text, tool_calls,
        reasoning_content: if reasoning.is_empty() { None } else { Some(reasoning) },
        finish_reason: stop_reason,
        usage,
        content_blocks,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::stream;
    use futures::StreamExt;
    use std::pin::Pin;

    type BytesResult = std::result::Result<bytes::Bytes, reqwest::Error>;

    #[tokio::test]
    async fn async_typed_event_stream_yields_events() {
        let raw = concat!(
            "data: {\"type\":\"message_start\",\"message\":{\"id\":\"m\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"c\"}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n",
            "data: {\"type\":\"message_stop\"}\n\n",
            "data: [DONE]\n\n"
        );
        let s: Pin<Box<dyn futures::Stream<Item = BytesResult> + Send>> = Box::pin(
            stream::once(async move { Ok(bytes::Bytes::from(raw)) }),
        );
        let mut st = AsyncStreamEventStream::new(s);
        let e1 = st.next().await.unwrap().unwrap();
        assert_eq!(e1.event_type, "message_start");
        let e2 = st.next().await.unwrap().unwrap();
        assert_eq!(e2.delta.as_ref().unwrap().text.as_deref(), Some("hi"));
        let e3 = st.next().await.unwrap().unwrap();
        assert_eq!(e3.event_type, "message_stop");
        assert!(st.next().await.is_none());
    }
}
