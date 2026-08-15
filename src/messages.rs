// Copyright (c) 2025 xiefujin <490021684@qq.com>
// Licensed under Apache-2.0, see LICENSE file for full license terms.

//! Messages API — synchronous and streaming.

use std::collections::BTreeMap;
use std::io::Read;
use std::sync::atomic::{AtomicBool, Ordering};

use serde_json::{json, Value};

use crate::client::AnthropicClient;
use crate::error::{AnthropicError, Result};
use crate::sse::SseReader;
use crate::types::{
    ChatMessage, LlmResponse, SimplifiedToolCall, Tool,
};

/// Accumulator for one streamed content block.
#[derive(Default, Clone)]
struct BlockAcc {
    kind: String,
    id: String,
    name: String,
    text: String,
    partial_json: String,
}

impl AnthropicClient {
    // ── Non-streaming messages ─────────────────────────────────────────────

    /// Create a message (non-streaming).
    pub fn messages_create(
        &self,
        messages: &[ChatMessage],
        system: Option<&str>,
        tools: Option<&[Tool]>,
        max_tokens: u64,
    ) -> Result<LlmResponse> {
        let body = self.build_message_body(messages, system, tools, max_tokens, false);
        let resp = self.post_json(body)?;
        let raw: Value = resp.into_json()?;
        Ok(self.assemble_message_response(&raw))
    }

    /// Create a message and return the raw JSON.
    pub fn messages_create_raw(
        &self,
        messages: &[ChatMessage],
        system: Option<&str>,
        tools: Option<&[Tool]>,
        max_tokens: u64,
    ) -> Result<Value> {
        let body = self.build_message_body(messages, system, tools, max_tokens, false);
        let resp = self.post_json(body)?;
        Ok(resp.into_json()?)
    }

    // ── Streaming messages (callback-based) ────────────────────────────────

    /// Stream a message with callbacks.
    ///
    /// `on_delta` is called for each text/reasoning token.
    /// `on_tool_call` is called when a tool call is completed.
    pub fn messages_stream(
        &self,
        messages: &[ChatMessage],
        system: Option<&str>,
        tools: Option<&[Tool]>,
        max_tokens: u64,
        on_delta: impl FnMut(&str),
        on_tool_call: impl FnMut(&str, &str),
    ) -> Result<LlmResponse> {
        let cancel = AtomicBool::new(false);
        let body = self.build_message_body(messages, system, tools, max_tokens, true);
        let reader = self.post_stream(body)?;
        parse_anthropic_stream(reader, on_delta, on_tool_call, &cancel)
    }

    /// Stream a message with cancellation support.
    pub fn messages_stream_cancellable(
        &self,
        messages: &[ChatMessage],
        system: Option<&str>,
        tools: Option<&[Tool]>,
        max_tokens: u64,
        on_delta: impl FnMut(&str),
        on_tool_call: impl FnMut(&str, &str),
        cancel: &AtomicBool,
    ) -> Result<LlmResponse> {
        let body = self.build_message_body(messages, system, tools, max_tokens, true);
        let reader = self.post_stream(body)?;
        parse_anthropic_stream(reader, on_delta, on_tool_call, cancel)
    }

    // ── Internal helpers ───────────────────────────────────────────────────

    fn build_message_body(
        &self,
        messages: &[ChatMessage],
        system: Option<&str>,
        tools: Option<&[Tool]>,
        max_tokens: u64,
        stream: bool,
    ) -> Value {
        let msgs = Self::build_anthropic_messages(messages);
        let mut body = json!({
            "model": self.model(),
            "max_tokens": max_tokens,
            "messages": msgs,
            "stream": stream,
        });

        if let Some(sys) = system {
            if !sys.is_empty() {
                body["system"] = Value::String(sys.to_string());
            }
        }

        if let Some(tools) = tools {
            if !tools.is_empty() {
                let arr: Vec<Value> = tools
                    .iter()
                    .map(|t| serde_json::to_value(t).unwrap_or_default())
                    .collect();
                body["tools"] = Value::Array(arr);
            }
        }

        body
    }

    /// Convert internal ChatMessages to Anthropic JSON format.
    fn build_anthropic_messages(messages: &[ChatMessage]) -> Vec<Value> {
        let mut out: Vec<Value> = Vec::new();
        let mut pending_tool_results: Vec<Value> = Vec::new();

        fn flush_tool_results(out: &mut Vec<Value>, pending: &mut Vec<Value>) {
            if !pending.is_empty() {
                out.push(json!({"role": "user", "content": std::mem::take(pending)}));
            }
        }

        for m in messages {
            // Any non-tool message ends a run of tool results.
            if m.role != "tool" {
                flush_tool_results(&mut out, &mut pending_tool_results);
            }

            match m.role.as_str() {
                "tool" => {
                    pending_tool_results.push(json!({
                        "type": "tool_result",
                        "tool_use_id": m.tool_call_id.clone().unwrap_or_default(),
                        "content": m.content,
                    }));
                }
                "assistant" => {
                    if let Some(tcs) = &m.tool_calls {
                        let mut blocks: Vec<Value> = Vec::new();
                        if !m.content.is_empty() {
                            blocks.push(json!({"type": "text", "text": m.content}));
                        }
                        for tc in tcs {
                            blocks.push(json!({
                                "type": "tool_use",
                                "id": tc.id,
                                "name": tc.name,
                                "input": tc.input,
                            }));
                        }
                        out.push(json!({"role": "assistant", "content": blocks}));
                    } else if !m.content.is_empty() {
                        out.push(if let Some(ref blocks) = m.content_blocks {
                            json!({"role": "assistant", "content": serde_json::to_value(blocks).unwrap_or_default()})
                        } else {
                            json!({"role": "assistant", "content": m.content})
                        });
                    }
                }
                _ => {
                    // user / system (embedded as user) / developer
                    if let Some(ref blocks) = m.content_blocks {
                        // Multi-block user message (e.g. text + image)
                        let content: Vec<Value> = blocks
                            .iter()
                            .map(|b| serde_json::to_value(b).unwrap_or_default())
                            .collect();
                        out.push(json!({"role": "user", "content": content}));
                    } else if !m.content.is_empty() {
                        out.push(json!({"role": "user", "content": m.content}));
                    }
                }
            }
        }

        // Flush trailing tool results.
        flush_tool_results(&mut out, &mut pending_tool_results);

        out
    }

    fn assemble_message_response(&self, raw: &Value) -> LlmResponse {
        let mut text = String::new();
        let mut tool_calls: Vec<SimplifiedToolCall> = Vec::new();
        let mut reasoning: Option<String> = None;
        let stop_reason = raw
            .get("stop_reason")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        if let Some(content) = raw.get("content").and_then(|v| v.as_array()) {
            for block in content {
                let btype = block.get("type").and_then(|v| v.as_str()).unwrap_or("");
                match btype {
                    "text" => {
                        if let Some(t) = block.get("text").and_then(|v| v.as_str()) {
                            text.push_str(t);
                        }
                    }
                    "thinking" => {
                        if let Some(t) = block.get("thinking").and_then(|v| v.as_str()) {
                            reasoning = Some(t.to_string());
                        }
                    }
                    "tool_use" => {
                        let id = block
                            .get("id")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let name = block
                            .get("name")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let input = block
                            .get("input")
                            .cloned()
                            .unwrap_or(json!({}));
                        let args = serde_json::to_string(&input).unwrap_or_default();
                        tool_calls.push(SimplifiedToolCall {
                            id,
                            name,
                            arguments: args,
                        });
                    }
                    _ => {}
                }
            }
        }

        let usage = raw.get("usage").map(|u| {
            serde_json::from_value(u.clone()).unwrap_or_default()
        });

        LlmResponse {
            text,
            tool_calls,
            reasoning_content: reasoning,
            finish_reason: stop_reason,
            usage,
        }
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
                return Err(AnthropicError::Api(format!("stream error: {msg}")));
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
                                acc.text.push_str(t);
                                on_delta(t);
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
            }
            _ => {}
        }
    }

    if total_payloads > 0 && valid_chunks == 0 {
        return Err(AnthropicError::Api(
            "stream returned no parseable data (all chunks malformed)".into(),
        ));
    }

    // No stop_reason but we got content → connection_closed.
    if stop_reason.is_none()
        && (!text.is_empty() || !reasoning.is_empty() || !blocks.is_empty())
    {
        stop_reason = Some("connection_closed".to_string());
    }

    // Assemble tool calls.
    let mut tool_calls = Vec::new();
    for (i, (_, acc)) in blocks.into_iter().enumerate() {
        if acc.kind != "tool_use" || acc.name.is_empty() {
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
            id,
            name: acc.name,
            arguments: args,
        });
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
        usage: None,
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
}