// Copyright (c) 2025 xiefujin <490021684@qq.com>
// Licensed under Apache-2.0, see LICENSE file for full license terms.

//! Shared API logic used by both sync and async clients.
//!
//! Message building, body construction, response assembly.

use serde_json::{json, Value};

use crate::types::{ChatMessage, LlmResponse, SimplifiedToolCall, Tool};

/// Build the request body for an Anthropic message.
pub fn build_message_body(
    model: &str,
    messages: &[ChatMessage],
    system: Option<&str>,
    tools: Option<&[Tool]>,
    max_tokens: u64,
    stream: bool,
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
                    let id = block.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
                    let name = block.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string();
                    let input = block.get("input").cloned().unwrap_or(json!({}));
                    let args = serde_json::to_string(&input).unwrap_or_default();
                    tool_calls.push(SimplifiedToolCall { id, name, arguments: args });
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
    }
}
