// Copyright (c) 2025 xiefujin <490021684@qq.com>
// Licensed under Apache-2.0, see LICENSE file for full license terms.

//! Shared API logic used by both sync and async clients.
//!
//! Message building, body construction, response assembly.

use serde_json::{json, Value};

use crate::request::ThinkingConfig;
use crate::types::{ChatMessage, ContentBlock, LlmResponse, SimplifiedToolCall, Tool};

/// Build the request body for an Anthropic message.
pub fn build_message_body(
    model: &str,
    messages: &[ChatMessage],
    system: Option<&str>,
    tools: Option<&[Tool]>,
    max_tokens: u64,
    stream: bool,
    thinking: Option<&ThinkingConfig>,
) -> Value {
    let msgs = build_anthropic_messages(messages);
    let mut body = json!({
        "model": model,
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
    if let Some(thinking) = thinking {
        body["thinking"] = serde_json::to_value(thinking).unwrap_or_default();
    }
    body
}

/// Build the request body for the `messages/count_tokens` endpoint.
pub fn build_count_tokens_body(
    model: &str,
    messages: &[ChatMessage],
    system: Option<&str>,
    tools: Option<&[Tool]>,
) -> Value {
    let msgs = build_anthropic_messages(messages);
    let mut body = json!({
        "model": model,
        "messages": msgs,
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
pub fn build_anthropic_messages(messages: &[ChatMessage]) -> Vec<Value> {
    let mut out: Vec<Value> = Vec::new();
    let mut pending_tool_results: Vec<Value> = Vec::new();

    fn flush_tool_results(out: &mut Vec<Value>, pending: &mut Vec<Value>) {
        if !pending.is_empty() {
            out.push(json!({"role": "user", "content": std::mem::take(pending)}));
        }
    }

    for m in messages {
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
                // Prefer explicit content blocks (text / thinking /
                // redacted_thinking) so a multi-turn conversation can pass
                // thinking blocks (with signatures) back, as Anthropic
                // requires for extended thinking + tool use. Fall back to
                // the plain-text `content` field otherwise.
                let mut blocks: Vec<Value> = Vec::new();
                if let Some(ref cbs) = m.content_blocks {
                    for cb in cbs {
                        blocks.push(serde_json::to_value(cb).unwrap_or_default());
                    }
                } else if !m.content.is_empty() {
                    blocks.push(json!({"type": "text", "text": m.content}));
                }
                if let Some(tcs) = &m.tool_calls {
                    for tc in tcs {
                        blocks.push(json!({
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.input,
                        }));
                    }
                }
                if !blocks.is_empty() {
                    out.push(json!({"role": "assistant", "content": blocks}));
                }
            }
            _ => {
                if let Some(ref blocks) = m.content_blocks {
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
    flush_tool_results(&mut out, &mut pending_tool_results);
    out
}

/// Assemble a message response into LlmResponse.
pub fn assemble_message_response(raw: &Value) -> LlmResponse {
    let mut text = String::new();
    let mut tool_calls: Vec<SimplifiedToolCall> = Vec::new();
    let mut reasoning: Option<String> = None;
    let mut content_blocks: Vec<ContentBlock> = Vec::new();
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
                        content_blocks.push(ContentBlock::Text {
                            text: t.to_string(),
                            citations: block
                                .get("citations")
                                .cloned()
                                .and_then(|c| serde_json::from_value(c).ok()),
                        });
                    }
                }
                "thinking" => {
                    if let Some(t) = block.get("thinking").and_then(|v| v.as_str()) {
                        reasoning = Some(t.to_string());
                        content_blocks.push(ContentBlock::Thinking {
                            thinking: t.to_string(),
                            signature: block
                                .get("signature")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string(),
                        });
                    }
                }
                "redacted_thinking" => {
                    // Opaque encrypted data; surface it so multi-turn
                    // conversations can pass it back unchanged.
                    let data = block
                        .get("data")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    content_blocks.push(ContentBlock::RedactedThinking { data });
                }
                "tool_use" => {
                    let id = block.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
                    let name = block.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string();
                    let input = block.get("input").cloned().unwrap_or(json!({}));
                    let args = serde_json::to_string(&input).unwrap_or_default();
                    tool_calls.push(SimplifiedToolCall { id: id.clone(), name: name.clone(), arguments: args });
                    content_blocks.push(ContentBlock::ToolUse { id, name, input });
                }
                _ => {}
            }
        }
    }
    let usage = raw.get("usage").map(|u| serde_json::from_value(u.clone()).unwrap_or_default());
    LlmResponse {
        text,
        tool_calls,
        reasoning_content: reasoning,
        finish_reason: stop_reason,
        usage,
        content_blocks,
    }
}
