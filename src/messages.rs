// Copyright (c) 2025 xiefujin <490021684@qq.com>
// Licensed under Apache-2.0, see LICENSE file for full license terms.

//! Messages API — synchronous and streaming.

use std::collections::BTreeMap;
use std::io::Read;
use std::sync::atomic::{AtomicBool, Ordering};

use serde_json::{json, Value};

use crate::client::AnthropicClient;
use crate::error::{AnthropicError, Result};
use crate::request::{MessageRequest, ThinkingConfig};
use crate::sse::SseReader;
use crate::types::{
    ChatMessage, ContentBlock, CountTokensResponse, LlmResponse, SimplifiedToolCall, StreamEvent,
    Tool, Usage,
};

/// Iterator over raw, typed [`StreamEvent`]s from a streamed response.
///
/// Stops at `data: [DONE]` / EOF. Convenience counterpart to the
/// callback-based [`AnthropicClient::messages_stream`].
pub struct StreamEventStream<R: Read> {
    sse: SseReader<R>,
}

impl<R: Read> StreamEventStream<R> {
    pub fn new(reader: R) -> Self {
        StreamEventStream { sse: SseReader::new(reader) }
    }
}

impl<R: Read> Iterator for StreamEventStream<R> {
    type Item = std::result::Result<StreamEvent, AnthropicError>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.sse.next_data() {
            Ok(Some(payload)) => Some(match serde_json::from_str(&payload) {
                Ok(ev) => Ok(ev),
                Err(e) => Err(AnthropicError::Json(e.to_string())),
            }),
            Ok(None) => None,
            Err(e) => Some(Err(AnthropicError::Network(format!("SSE read error: {e}")))),
        }
    }
}

/// Accumulator for one streamed content block.
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

impl AnthropicClient {
    // ── Non-streaming messages ─────────────────────────────────────────────

    /// Create a message (non-streaming).
    ///
    /// `thinking` enables extended thinking (see [`ThinkingConfig`]).
    pub fn messages_create(
        &self,
        messages: &[ChatMessage],
        system: Option<&str>,
        tools: Option<&[Tool]>,
        max_tokens: u64,
        thinking: Option<&ThinkingConfig>,
    ) -> Result<LlmResponse> {
        let body = crate::api_common::build_message_body(
            self.model(),
            messages,
            system,
            tools,
            max_tokens,
            false,
            thinking,
        );
        let resp = self.post_json(body)?;
        let raw: Value = resp.into_json()?;
        Ok(crate::api_common::assemble_message_response(&raw))
    }

    /// Create a message and return the raw JSON.
    pub fn messages_create_raw(
        &self,
        messages: &[ChatMessage],
        system: Option<&str>,
        tools: Option<&[Tool]>,
        max_tokens: u64,
        thinking: Option<&ThinkingConfig>,
    ) -> Result<Value> {
        let body = crate::api_common::build_message_body(
            self.model(),
            messages,
            system,
            tools,
            max_tokens,
            false,
            thinking,
        );
        let resp = self.post_json(body)?;
        Ok(resp.into_json()?)
    }

    /// Count input tokens for a set of messages via the official
    /// `messages/count_tokens` endpoint.
    pub fn count_tokens(
        &self,
        messages: &[ChatMessage],
        system: Option<&str>,
        tools: Option<&[Tool]>,
    ) -> Result<CountTokensResponse> {
        let body = crate::api_common::build_count_tokens_body(self.model(), messages, system, tools);
        let resp = self.post_json_path("v1/messages/count_tokens", body)?;
        Ok(resp.into_json()?)
    }

    /// Stream and return raw typed [`StreamEvent`]s.
    pub fn messages_stream_events(
        &self,
        messages: &[ChatMessage],
        system: Option<&str>,
        tools: Option<&[Tool]>,
        max_tokens: u64,
        thinking: Option<&ThinkingConfig>,
    ) -> Result<StreamEventStream<Box<dyn Read>>> {
        let body = crate::api_common::build_message_body(
            self.model(),
            messages,
            system,
            tools,
            max_tokens,
            true,
            thinking,
        );
        let reader = self.post_stream(body)?;
        Ok(StreamEventStream::new(Box::new(reader) as Box<dyn Read>))
    }

    // ── Streaming messages (callback-based) ────────────────────────────────

    /// Stream a message with callbacks.
    ///
    /// `on_delta` is called for each text/reasoning token.
    /// `on_tool_call` is called when a tool call is completed.
    #[allow(clippy::too_many_arguments)]
    pub fn messages_stream(
        &self,
        messages: &[ChatMessage],
        system: Option<&str>,
        tools: Option<&[Tool]>,
        max_tokens: u64,
        thinking: Option<&ThinkingConfig>,
        on_delta: impl FnMut(&str),
        on_tool_call: impl FnMut(&str, &str),
    ) -> Result<LlmResponse> {
        let cancel = AtomicBool::new(false);
        let body = crate::api_common::build_message_body(
            self.model(),
            messages,
            system,
            tools,
            max_tokens,
            true,
            thinking,
        );
        let reader = self.post_stream(body)?;
        parse_anthropic_stream(reader, on_delta, on_tool_call, &cancel)
    }

    /// Stream a message with cancellation support.
    #[allow(clippy::too_many_arguments)]
    pub fn messages_stream_cancellable(
        &self,
        messages: &[ChatMessage],
        system: Option<&str>,
        tools: Option<&[Tool]>,
        max_tokens: u64,
        thinking: Option<&ThinkingConfig>,
        on_delta: impl FnMut(&str),
        on_tool_call: impl FnMut(&str, &str),
        cancel: &AtomicBool,
    ) -> Result<LlmResponse> {
        let body = crate::api_common::build_message_body(
            self.model(),
            messages,
            system,
            tools,
            max_tokens,
            true,
            thinking,
        );
        let reader = self.post_stream(body)?;
        parse_anthropic_stream(reader, on_delta, on_tool_call, cancel)
    }

    // ── Full request-builder API ───────────────────────────────────────────

    /// Send a [`MessageRequest`] (non-streaming).
    ///
    /// Unlike the convenience methods this exposes every Messages API
    /// parameter (temperature, top_p, top_k, stop_sequences, thinking,
    /// metadata, tool_choice, ...).
    pub fn messages_send(&self, request: &MessageRequest) -> Result<LlmResponse> {
        let mut body = request.build_body();
        body["stream"] = json!(false);
        let resp = self.post_json(body)?;
        let raw: Value = resp.into_json()?;
        Ok(crate::api_common::assemble_message_response(&raw))
    }

    /// Send a [`MessageRequest`] with streaming.
    pub fn messages_send_stream(
        &self,
        request: &MessageRequest,
        on_delta: impl FnMut(&str),
        on_tool_call: impl FnMut(&str, &str),
    ) -> Result<LlmResponse> {
        let cancel = AtomicBool::new(false);
        let mut body = request.build_body();
        body["stream"] = json!(true);
        let reader = self.post_stream(body)?;
        parse_anthropic_stream(reader, on_delta, on_tool_call, &cancel)
    }

    /// Send a [`MessageRequest`] with streaming + cancellation.
    pub fn messages_send_stream_cancellable(
        &self,
        request: &MessageRequest,
        on_delta: impl FnMut(&str),
        on_tool_call: impl FnMut(&str, &str),
        cancel: &AtomicBool,
    ) -> Result<LlmResponse> {
        let mut body = request.build_body();
        body["stream"] = json!(true);
        let reader = self.post_stream(body)?;
        parse_anthropic_stream(reader, on_delta, on_tool_call, cancel)
    }

    /// Convert internal ChatMessages to Anthropic JSON format.
    pub fn build_anthropic_messages(messages: &[ChatMessage]) -> Vec<Value> {
        crate::api_common::build_anthropic_messages(messages)
    }
}

/// Parse an Anthropic SSE stream from any reader.
pub fn parse_anthropic_stream<R: Read>(
    reader: R,
    mut on_delta: impl FnMut(&str),
    mut on_tool_call: impl FnMut(&str, &str),
    cancel: &AtomicBool,
) -> Result<LlmResponse> {
    let mut sse = SseReader::new(reader);
    let mut blocks: BTreeMap<i64, BlockAcc> = BTreeMap::new();
    let mut text = String::new();
    let mut reasoning = String::new();
    let mut stop_reason: Option<String> = None;
    let mut usage: Option<Usage> = None;
    let mut valid_chunks: usize = 0;
    let mut total_payloads: usize = 0;

    while let Some(payload) = match sse.next_data() {
        Ok(Some(p)) => Some(p),
        Ok(None) => None,
        Err(e) => return Err(AnthropicError::Network(format!("SSE read error: {e}"))),
    } {
        if cancel.load(Ordering::SeqCst) {
            return Err(AnthropicError::Cancelled);
        }
        total_payloads += 1;
        let ev: Value = match serde_json::from_str(&payload) {
            Ok(v) => {
                valid_chunks += 1;
                v
            }
            Err(_) => continue,
        };

        let etype = ev.get("type").and_then(|v| v.as_str()).unwrap_or("");
        match etype {
            "error" => {
                let msg = ev
                    .get("error")
                    .and_then(|e| e.get("message"))
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| ev.to_string());
                return Err(AnthropicError::stream_error(format!("stream error: {msg}")));
            }
            "message_start" => {
                // The `message_start` event carries input usage up front;
                // `message_delta` later carries the full cumulative usage.
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
                    acc.kind = b
                        .get("type")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    acc.id = b
                        .get("id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    acc.name = b
                        .get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    // redacted_thinking blocks arrive fully-formed here.
                    acc.data = b
                        .get("data")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
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
                                text.push_str(t);
                                acc.text.push_str(t);
                                on_delta(t);
                            }
                        }
                        "thinking_delta" => {
                            if let Some(t) = d.get("thinking").and_then(|v| v.as_str()) {
                                reasoning.push_str(t);
                                acc.thinking.push_str(t);
                                on_delta(t);
                            }
                        }
                        // Closes a thinking block; the signature must be
                        // kept so the block can be passed back in the next
                        // turn of a multi-turn conversation.
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
                if let Some(sr) = ev
                    .get("delta")
                    .and_then(|d| d.get("stop_reason"))
                    .and_then(|v| v.as_str())
                {
                    stop_reason = Some(sr.to_string());
                }
                // Full cumulative token usage arrives with message_delta.
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
        return Err(AnthropicError::stream_error(
            "stream returned no parseable data (all chunks malformed)",
        ));
    }

    // No stop_reason but we got content → connection_closed.
    if stop_reason.is_none()
        && (!text.is_empty() || !reasoning.is_empty() || !blocks.is_empty())
    {
        stop_reason = Some("connection_closed".to_string());
    }

    // Assemble tool calls and ordered content blocks (in stream index order).
    let mut tool_calls = Vec::new();
    let mut content_blocks: Vec<ContentBlock> = Vec::new();
    for (i, (_, acc)) in blocks.into_iter().enumerate() {
        match acc.kind.as_str() {
            "text" => {
                if !acc.text.is_empty() {
                    content_blocks.push(ContentBlock::Text {
                        text: acc.text,
                        citations: None,
                    });
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
                if acc.name.is_empty() {
                    continue;
                }
                let args = if acc.partial_json.trim().is_empty() {
                    "{}".to_string()
                } else {
                    acc.partial_json
                };
                let id = if acc.id.is_empty() {
                    format!("call_{i}")
                } else {
                    acc.id
                };
                on_tool_call(&acc.name, &args);
                tool_calls.push(SimplifiedToolCall {
                    id: id.clone(),
                    name: acc.name.clone(),
                    arguments: args.clone(),
                });
                content_blocks.push(ContentBlock::ToolUse {
                    id,
                    name: acc.name,
                    input: serde_json::from_str(&args).unwrap_or_else(|_| json!({})),
                });
            }
            _ => {}
        }
    }

    if matches!(stop_reason.as_deref(), Some("max_tokens")) {
        text.push_str(
            "\n\n[WARNING: API response truncated (max_tokens). Reduce content or raise max_tokens.]",
        );
    }

    Ok(LlmResponse {
        text,
        tool_calls,
        reasoning_content: if reasoning.is_empty() {
            None
        } else {
            Some(reasoning)
        },
        finish_reason: stop_reason,
        usage,
        content_blocks,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use crate::types::ToolCall as FullToolCall;

    #[test]
    fn build_messages_coalesces_parallel_tool_results() {
        let _client = AnthropicClient::new("sk-ant-test", "test");
        let msgs = vec![
            ChatMessage::user("do two things"),
            ChatMessage::assistant_with_tools(
                "",
                vec![
                    FullToolCall {
                        id: "a".into(),
                        call_type: "tool_use".into(),
                        name: "run_shell".into(),
                        input: json!({"command": "cat x"}),
                    },
                    FullToolCall {
                        id: "b".into(),
                        call_type: "tool_use".into(),
                        name: "run_shell".into(),
                        input: json!({"command": "python3 x"}),
                    },
                ],
            ),
            ChatMessage::tool_result("a", "content of x"),
            ChatMessage::tool_result("b", "ran x"),
            ChatMessage::user("thanks"),
        ];
        let out = AnthropicClient::build_anthropic_messages(&msgs);
        // out[0]=user, out[1]=assistant(2 tool_use), out[2]=user(2 tool_result), out[3]=user
        assert_eq!(out[1]["content"].as_array().unwrap().len(), 2);
        let tool_result_msg = &out[2];
        assert_eq!(tool_result_msg["role"], "user");
        let blocks = tool_result_msg["content"].as_array().unwrap();
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0]["type"], "tool_result");
        assert_eq!(blocks[0]["tool_use_id"], "a");
        assert_eq!(blocks[1]["tool_use_id"], "b");
        assert_eq!(out[3]["role"], "user");
        assert_eq!(out[3]["content"], "thanks");
    }

    #[test]
    fn stream_parses_text_and_thinking() {
        let raw = concat!(
            "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\"}}\n\n",
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"reason\"}}\n\n",
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n",
            "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n",
            "data: [DONE]\n\n"
        );
        let cancel = AtomicBool::new(false);
        let resp = parse_anthropic_stream(
            Cursor::new(raw.as_bytes().to_vec()),
            |_| {},
            |_, _| {},
            &cancel,
        )
        .unwrap();
        assert_eq!(resp.text, "hi");
        assert_eq!(resp.reasoning_content.as_deref(), Some("reason"));
    }

    #[test]
    fn stream_parses_tool_use() {
        let raw = concat!(
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"t1\",\"name\":\"run_shell\"}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"command\\\":\\\"\"}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"ls\\\"}\"}}\n\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"}}\n\n",
            "data: [DONE]\n\n"
        );
        let cancel = AtomicBool::new(false);
        let mut tools = Vec::new();
        let resp = parse_anthropic_stream(
            Cursor::new(raw.as_bytes().to_vec()),
            |_| {},
            |n, a| tools.push((n.to_string(), a.to_string())),
            &cancel,
        )
        .unwrap();
        assert_eq!(resp.tool_calls.len(), 1);
        assert_eq!(resp.tool_calls[0].name, "run_shell");
        assert_eq!(resp.tool_calls[0].id, "t1");
        assert_eq!(resp.tool_calls[0].parsed_args()["command"], "ls");
    }

    #[test]
    fn stream_parses_signature_delta_and_builds_ordered_blocks() {
        let raw = concat!(
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\"}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"Let me think\"}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"sig_abc\"}}\n\n",
            "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
            "data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"text\"}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"answer\"}}\n\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"input_tokens\":10,\"output_tokens\":5}}\n\n",
            "data: [DONE]\n\n"
        );
        let cancel = AtomicBool::new(false);
        let resp = parse_anthropic_stream(
            Cursor::new(raw.as_bytes().to_vec()),
            |_| {},
            |_, _| {},
            &cancel,
        )
        .unwrap();
        // reasoning accumulated, text clean
        assert_eq!(resp.reasoning_content.as_deref(), Some("Let me think"));
        assert_eq!(resp.text, "answer");
        // ordered content blocks: thinking (with signature) then text
        assert_eq!(resp.content_blocks.len(), 2);
        match &resp.content_blocks[0] {
            ContentBlock::Thinking { thinking, signature } => {
                assert_eq!(thinking, "Let me think");
                assert_eq!(signature, "sig_abc");
            }
            other => panic!("expected Thinking, got {other:?}"),
        }
        match &resp.content_blocks[1] {
            ContentBlock::Text { text, .. } => assert_eq!(text, "answer"),
            other => panic!("expected Text, got {other:?}"),
        }
        // usage captured from message_delta
        let u = resp.usage.unwrap();
        assert_eq!(u.input_tokens, 10);
        assert_eq!(u.output_tokens, 5);
    }

    #[test]
    fn stream_parses_redacted_thinking_block() {
        let raw = concat!(
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"redacted_thinking\",\"data\":\"encrypted==\"}}\n\n",
            "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n",
            "data: [DONE]\n\n"
        );
        let cancel = AtomicBool::new(false);
        let resp = parse_anthropic_stream(
            Cursor::new(raw.as_bytes().to_vec()),
            |_| {},
            |_, _| {},
            &cancel,
        )
        .unwrap();
        assert_eq!(resp.content_blocks.len(), 1);
        match &resp.content_blocks[0] {
            ContentBlock::RedactedThinking { data } => assert_eq!(data, "encrypted=="),
            other => panic!("expected RedactedThinking, got {other:?}"),
        }
        assert!(resp.reasoning_content.is_none());
    }

    #[test]
    fn stream_roundtrips_thinking_block_via_build_messages() {
        // A thinking block captured from a stream (with signature) must
        // serialize back into the assistant message for the next turn.
        let blocks = vec![
            ContentBlock::Thinking {
                thinking: "Let me think".into(),
                signature: "sig_abc".into(),
            },
            ContentBlock::ToolUse {
                id: "t1".into(),
                name: "run".into(),
                input: json!({"cmd": "ls"}),
            },
        ];
        let mut msg = ChatMessage::assistant("");
        msg.content_blocks = Some(blocks);
        let out = AnthropicClient::build_anthropic_messages(&[msg]);
        let content = out[0]["content"].as_array().unwrap();
        assert_eq!(content.len(), 2);
        assert_eq!(content[0]["type"], "thinking");
        assert_eq!(content[0]["signature"], "sig_abc");
        assert_eq!(content[1]["type"], "tool_use");
    }

    #[test]
    fn thinking_delta_not_leaked_into_text_blocks() {
        let raw = concat!(
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\"}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"reason\"}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"sig_x\"}}\n\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n",
            "data: [DONE]\n\n"
        );
        let cancel = AtomicBool::new(false);
        let resp = parse_anthropic_stream(
            Cursor::new(raw.as_bytes().to_vec()),
            |_| {},
            |_, _| {},
            &cancel,
        )
        .unwrap();
        assert_eq!(resp.text, "");
        assert_eq!(resp.reasoning_content.as_deref(), Some("reason"));
        // The thinking content must NOT appear in any Text block.
        match &resp.content_blocks[0] {
            ContentBlock::Thinking { thinking, .. } => assert_eq!(thinking, "reason"),
            other => panic!("expected Thinking, got {other:?}"),
        }
    }

    #[test]
    fn stream_max_tokens_warning() {
        let raw = concat!(
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\"}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"p\"}}\n\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"max_tokens\"}}\n\n",
            "data: [DONE]\n\n"
        );
        let cancel = AtomicBool::new(false);
        let resp = parse_anthropic_stream(
            Cursor::new(raw.as_bytes().to_vec()),
            |_| {},
            |_, _| {},
            &cancel,
        )
        .unwrap();
        assert!(resp.is_truncated());
        assert!(resp.text.contains("truncated"));
    }

    #[test]
    fn sse_read_error_propagates() {
        struct BrokenReader;
        impl Read for BrokenReader {
            fn read(&mut self, _buf: &mut [u8]) -> std::io::Result<usize> {
                Err(std::io::Error::new(
                    std::io::ErrorKind::TimedOut,
                    "simulated timeout",
                ))
            }
        }
        let cancel = AtomicBool::new(false);
        let r = parse_anthropic_stream(BrokenReader, |_| {}, |_, _| {}, &cancel);
        assert!(r.is_err());
    }

    #[test]
    fn stream_no_stop_reason_gets_connection_closed() {
        let raw = concat!(
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\"}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"content\"}}\n\n"
        );
        let cancel = AtomicBool::new(false);
        let resp = parse_anthropic_stream(
            Cursor::new(raw.as_bytes().to_vec()),
            |_| {},
            |_, _| {},
            &cancel,
        )
        .unwrap();
        assert_eq!(
            resp.finish_reason,
            Some("connection_closed".to_string())
        );
        assert!(resp.is_truncated());
    }

    #[test]
    fn all_malformed_chunks_error() {
        let raw = concat!(
            "data: {not valid json\n\n",
            "data: }also not json\n\n",
            "data: [DONE]\n\n"
        );
        let cancel = AtomicBool::new(false);
        let r = parse_anthropic_stream(
            Cursor::new(raw.as_bytes().to_vec()),
            |_| {},
            |_, _| {},
            &cancel,
        );
        assert!(r.is_err());
    }

    #[test]
    fn typed_event_stream_yields_events() {
        let raw = concat!(
            "data: {\"type\":\"message_start\",\"message\":{\"id\":\"m\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"c\"}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n",
            "data: {\"type\":\"message_stop\"}\n\n",
            "data: [DONE]\n\n"
        );
        let mut it = StreamEventStream::new(Cursor::new(raw.as_bytes().to_vec()));
        let e1 = it.next().unwrap().unwrap();
        assert_eq!(e1.event_type, "message_start");
        let e2 = it.next().unwrap().unwrap();
        assert_eq!(e2.delta.as_ref().unwrap().text.as_deref(), Some("hi"));
        let e3 = it.next().unwrap().unwrap();
        assert_eq!(e3.event_type, "message_stop");
        assert!(it.next().is_none());
    }

    #[test]
    fn typed_event_stream_propagates_parse_error() {
        let raw = "data: {not json\n\ndata: [DONE]\n\n";
        let mut it = StreamEventStream::new(Cursor::new(raw.as_bytes().to_vec()));
        assert!(it.next().unwrap().is_err());
    }
}
