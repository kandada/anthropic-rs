// Copyright (c) 2025 xiefujin <490021684@qq.com>
// Licensed under Apache-2.0, see LICENSE file for full license terms.

//! Integration tests for anthropic-rs.
//!
//! Covers: types serde, request building, SSE parsing, tool use,
//! error handling, content blocks, and client configuration.

use serde_json::json;
use anthropic_client_rs::*;

// ── Types: serialization roundtrips ────────────────────────────────────────

#[test]
fn test_chat_message_serde_roundtrip() {
    let msg = ChatMessage::user("Hello world");
    let json_str = serde_json::to_string(&msg).unwrap();
    let decoded: ChatMessage = serde_json::from_str(&json_str).unwrap();
    assert_eq!(decoded.role, "user");
    assert_eq!(decoded.content, "Hello world");
}

#[test]
fn test_tool_call_serde_roundtrip() {
    let tc = ToolCall {
        id: "toolu_1".into(),
        call_type: "tool_use".into(),
        name: "get_weather".into(),
        input: json!({"location": "Paris"}),
    };
    let msg = ChatMessage::assistant_with_tools("", vec![tc]);
    let json_str = serde_json::to_string(&msg).unwrap();
    let decoded: ChatMessage = serde_json::from_str(&json_str).unwrap();
    let tcs = decoded.tool_calls.unwrap();
    assert_eq!(tcs[0].name, "get_weather");
    assert_eq!(tcs[0].input["location"], "Paris");
}

#[test]
fn test_content_block_text_serde() {
    let block = ContentBlock::Text { text: "Hello".into(), citations: None };
    let v = serde_json::to_value(&block).unwrap();
    assert_eq!(v["type"], "text");
    assert_eq!(v["text"], "Hello");

    let decoded: ContentBlock = serde_json::from_value(v).unwrap();
    match decoded {
        ContentBlock::Text { text, .. } => assert_eq!(text, "Hello"),
        _ => panic!("expected Text"),
    }
}

#[test]
fn test_content_block_tool_use_serde() {
    let block = ContentBlock::ToolUse {
        id: "toolu_1".into(),
        name: "get_weather".into(),
        input: json!({"location": "Tokyo"}),
    };
    let v = serde_json::to_value(&block).unwrap();
    assert_eq!(v["type"], "tool_use");
    assert_eq!(v["name"], "get_weather");
    assert_eq!(v["input"]["location"], "Tokyo");
}

#[test]
fn test_content_block_tool_result_serde() {
    let block = ContentBlock::ToolResult {
        tool_use_id: "toolu_1".into(),
        content: "Sunny, 25C".into(),
        is_error: None,
    };
    let v = serde_json::to_value(&block).unwrap();
    assert_eq!(v["type"], "tool_result");
    assert_eq!(v["tool_use_id"], "toolu_1");
    assert_eq!(v["content"], "Sunny, 25C");
}

#[test]
fn test_content_block_thinking_serde() {
    let block = ContentBlock::Thinking {
        thinking: "Let me think...".into(),
        signature: "sig_xxx".into(),
    };
    let v = serde_json::to_value(&block).unwrap();
    assert_eq!(v["type"], "thinking");
    assert_eq!(v["thinking"], "Let me think...");
    assert_eq!(v["signature"], "sig_xxx");
}

#[test]
fn test_content_block_image_serde() {
    let block = ContentBlock::Image {
        source: ImageSource {
            source_type: "base64".into(),
            media_type: "image/jpeg".into(),
            data: "base64data".into(),
        },
    };
    let v = serde_json::to_value(&block).unwrap();
    assert_eq!(v["type"], "image");
    assert_eq!(v["source"]["type"], "base64");
    assert_eq!(v["source"]["media_type"], "image/jpeg");
}

#[test]
fn test_message_response_deserialize() {
    let raw = json!({
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Hello!"},
            {"type": "tool_use", "id": "tu_1", "name": "run", "input": {"cmd": "ls"}}
        ],
        "model": "claude-sonnet-4-20250514",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5}
    });
    let mr: MessageResponse = serde_json::from_value(raw).unwrap();
    assert_eq!(mr.id, "msg_123");
    assert_eq!(mr.content.len(), 2);
    assert_eq!(mr.stop_reason.as_deref(), Some("end_turn"));
}

#[test]
fn test_tool_definition_serde() {
    let tool = Tool::new("get_weather", "Get weather", json!({
        "type": "object",
        "properties": {"location": {"type": "string"}},
        "required": ["location"]
    }));
    let v = serde_json::to_value(&tool).unwrap();
    assert_eq!(v["name"], "get_weather");
    assert_eq!(v["description"], "Get weather");
    assert!(v["input_schema"]["required"].as_array().unwrap().contains(&json!("location")));
}

#[test]
fn test_tool_choice_serde() {
    let auto = serde_json::to_value(&ToolChoice::auto()).unwrap();
    assert_eq!(auto["type"], "auto");

    let any = serde_json::to_value(&ToolChoice::any()).unwrap();
    assert_eq!(any["type"], "any");

    let specific = serde_json::to_value(&ToolChoice::specific("my_tool")).unwrap();
    assert_eq!(specific["type"], "tool");
    assert_eq!(specific["name"], "my_tool");
}

// ── Request builder ─────────────────────────────────────────────────────────

#[test]
fn test_request_builder_minimal() {
    let req = MessageRequest::new("claude-sonnet-4-20250514", vec![ChatMessage::user("hi")], 1024);
    let body = req.build_body();
    assert_eq!(body["model"], "claude-sonnet-4-20250514");
    assert_eq!(body["max_tokens"], 1024);
    assert_eq!(body["stream"], false);
    assert!(body.get("system").is_none());
    assert!(!body.as_object().unwrap().contains_key("temperature"));
}

#[test]
fn test_request_builder_full() {
    let tools = vec![Tool::new("fn", "desc", json!({"type":"object","properties":{}}))];
    let req = MessageRequest::new("claude-sonnet-4-20250514", vec![ChatMessage::user("hi")], 2048)
        .system("You are helpful.")
        .temperature(0.7)
        .top_p(0.9)
        .top_k(40)
        .stop_sequences(vec!["END".into(), "STOP".into()])
        .tools(tools)
        .tool_choice(ToolChoice::any())
        .thinking(ThinkingConfig::enabled(4096))
        .metadata(Metadata { user_id: Some("user-1".into()) });

    let body = req.build_body();
    assert_eq!(body["system"], "You are helpful.");
    assert_eq!(body["temperature"], 0.7);
    assert_eq!(body["top_p"], 0.9);
    assert_eq!(body["top_k"], 40);
    assert_eq!(body["stop_sequences"].as_array().unwrap().len(), 2);
    assert!(body["tools"].is_array());
    assert_eq!(body["tool_choice"]["type"], "any");
    assert_eq!(body["thinking"]["type"], "enabled");
    assert_eq!(body["thinking"]["budget_tokens"], 4096);
    assert_eq!(body["metadata"]["user_id"], "user-1");
}

#[test]
fn test_request_builder_empty_system_omitted() {
    let req = MessageRequest::new("claude-sonnet-4-20250514", vec![ChatMessage::user("hi")], 1024)
        .system("");
    let body = req.build_body();
    assert!(body.get("system").is_none());
}

#[test]
fn test_request_builder_stream() {
    let req = MessageRequest::new("claude-sonnet-4-20250514", vec![ChatMessage::user("hi")], 1024)
        .stream(true);
    let body = req.build_body();
    assert_eq!(body["stream"], true);
}

// ── SSE parsing (sync) ──────────────────────────────────────────────────────

#[test]
fn test_sse_parse_text_and_thinking() {
    use std::io::Cursor;
    use std::sync::atomic::AtomicBool;

    let raw = concat!(
        "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\"}}\n\n",
        "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"reason\"}}\n\n",
        "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n",
        "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n",
        "data: [DONE]\n\n"
    );
    let cancel = AtomicBool::new(false);
    let resp = anthropic_client_rs::parse_anthropic_stream(
        Cursor::new(raw.as_bytes().to_vec()),
        |_| {}, |_, _| {}, &cancel,
    ).unwrap();
    assert_eq!(resp.text, "hi");
    assert_eq!(resp.reasoning_content.as_deref(), Some("reason"));
}

#[test]
fn test_sse_parse_tool_use() {
    use std::io::Cursor;
    use std::sync::atomic::AtomicBool;

    let raw = concat!(
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"t1\",\"name\":\"run_shell\"}}\n\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"command\\\":\\\"\"}}\n\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"ls\\\"}\"}}\n\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"}}\n\n",
        "data: [DONE]\n\n"
    );
    let cancel = AtomicBool::new(false);
    let mut tools = Vec::new();
    let resp = anthropic_client_rs::parse_anthropic_stream(
        Cursor::new(raw.as_bytes().to_vec()),
        |_| {},
        |n, a| tools.push((n.to_string(), a.to_string())),
        &cancel,
    ).unwrap();
    assert_eq!(resp.tool_calls.len(), 1);
    assert_eq!(resp.tool_calls[0].name, "run_shell");
    assert_eq!(resp.tool_calls[0].id, "t1");
    assert_eq!(resp.tool_calls[0].parsed_args()["command"], "ls");
}

#[test]
fn test_sse_parse_max_tokens_warning() {
    use std::io::Cursor;
    use std::sync::atomic::AtomicBool;

    let raw = concat!(
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\"}}\n\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"p\"}}\n\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"max_tokens\"}}\n\n",
        "data: [DONE]\n\n"
    );
    let cancel = AtomicBool::new(false);
    let resp = anthropic_client_rs::parse_anthropic_stream(
        Cursor::new(raw.as_bytes().to_vec()),
        |_| {}, |_, _| {}, &cancel,
    ).unwrap();
    assert!(resp.is_truncated());
    assert!(resp.text.contains("truncated"));
}

#[test]
fn test_sse_parse_no_stop_reason_gets_connection_closed() {
    use std::io::Cursor;
    use std::sync::atomic::AtomicBool;

    let raw = concat!(
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\"}}\n\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"content\"}}\n\n"
    );
    let cancel = AtomicBool::new(false);
    let resp = anthropic_client_rs::parse_anthropic_stream(
        Cursor::new(raw.as_bytes().to_vec()),
        |_| {}, |_, _| {}, &cancel,
    ).unwrap();
    assert_eq!(resp.finish_reason.as_deref(), Some("connection_closed"));
    assert!(resp.is_truncated());
}

#[test]
fn test_sse_parse_all_malformed_error() {
    use std::io::Cursor;
    use std::sync::atomic::AtomicBool;

    let raw = concat!("data: {not json\n\n", "data: [DONE]\n\n");
    let cancel = AtomicBool::new(false);
    let r = anthropic_client_rs::parse_anthropic_stream(
        Cursor::new(raw.as_bytes().to_vec()),
        |_| {}, |_, _| {}, &cancel,
    );
    assert!(r.is_err());
}

#[test]
fn test_sse_parse_error_event() {
    use std::io::Cursor;
    use std::sync::atomic::AtomicBool;

    let raw = "data: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}\n\n";
    let cancel = AtomicBool::new(false);
    let r = anthropic_client_rs::parse_anthropic_stream(
        Cursor::new(raw.as_bytes().to_vec()),
        |_| {}, |_, _| {}, &cancel,
    );
    assert!(r.is_err());
    if let Err(AnthropicError::Api(e)) = r {
        assert!(e.message.contains("Overloaded"));
    } else {
        panic!("expected Api error");
    }
}

// ── Message building ────────────────────────────────────────────────────────

#[test]
fn test_build_anthropic_messages_coalesces_tool_results() {
    use anthropic_client_rs::api_common::build_anthropic_messages;

    let msgs = vec![
        ChatMessage::user("do things"),
        ChatMessage::assistant_with_tools(
            "",
            vec![
                ToolCall { id: "a".into(), call_type: "tool_use".into(), name: "run".into(), input: json!({"cmd":"x"}) },
                ToolCall { id: "b".into(), call_type: "tool_use".into(), name: "run".into(), input: json!({"cmd":"y"}) },
            ],
        ),
        ChatMessage::tool_result("a", "result a"),
        ChatMessage::tool_result("b", "result b"),
        ChatMessage::user("thanks"),
    ];

    let out = build_anthropic_messages(&msgs);
    // out[0]=user, out[1]=assistant(2 tool_use), out[2]=user(2 tool_result), out[3]=user("thanks")
    assert_eq!(out[1]["role"], "assistant");
    assert_eq!(out[1]["content"].as_array().unwrap().len(), 2);

    let tr_msg = &out[2];
    assert_eq!(tr_msg["role"], "user");
    let blocks = tr_msg["content"].as_array().unwrap();
    assert_eq!(blocks.len(), 2);
    assert_eq!(blocks[0]["type"], "tool_result");
    assert_eq!(blocks[0]["tool_use_id"], "a");
    assert_eq!(blocks[1]["tool_use_id"], "b");

    assert_eq!(out[3]["role"], "user");
    assert_eq!(out[3]["content"], "thanks");
}

#[test]
fn test_build_anthropic_messages_content_blocks() {
    use anthropic_client_rs::api_common::build_anthropic_messages;

    let msgs = vec![ChatMessage::user_with_blocks(vec![
        ContentBlock::Text { text: "Look at this".into(), citations: None },
        ContentBlock::Image {
            source: ImageSource {
                source_type: "url".into(),
                media_type: "image/jpeg".into(),
                data: "https://example.com/img.jpg".into(),
            },
        },
    ])];

    let out = build_anthropic_messages(&msgs);
    let content = out[0]["content"].as_array().unwrap();
    assert_eq!(content.len(), 2);
    assert_eq!(content[0]["type"], "text");
    assert_eq!(content[1]["type"], "image");
}

// ── Error handling ──────────────────────────────────────────────────────────

#[test]
fn test_error_retryable() {
    assert!(AnthropicError::Network("timeout".into()).is_retryable());
    assert!(AnthropicError::api(503, None, "Service Unavailable").is_retryable());
    assert!(AnthropicError::stream_error("rate limit exceeded").is_retryable());
    assert!(!AnthropicError::api(401, None, "Unauthorized").is_retryable());
    assert!(!AnthropicError::Config("no API key".into()).is_retryable());
}

#[test]
fn test_error_display() {
    let e = AnthropicError::Config("missing key".into());
    assert!(e.to_string().contains("config"));
    assert!(e.to_string().contains("missing key"));
    assert_eq!(AnthropicError::Cancelled.to_string(), "cancelled");
}

// ── LlmResponse ─────────────────────────────────────────────────────────────

#[test]
fn test_llm_response_truncated() {
    let resp = LlmResponse { finish_reason: Some("max_tokens".into()), ..Default::default() };
    assert!(resp.is_truncated());
    let resp = LlmResponse { finish_reason: Some("connection_closed".into()), ..Default::default() };
    assert!(resp.is_truncated());
    let resp = LlmResponse { finish_reason: Some("end_turn".into()), ..Default::default() };
    assert!(!resp.is_truncated());
}

// ── Client configuration ────────────────────────────────────────────────────

#[test]
fn test_client_new() {
    let c = AnthropicClient::new("sk-ant-test123", "claude-sonnet-4-20250514");
    assert_eq!(c.model(), "claude-sonnet-4-20250514");
}

#[test]
fn test_client_with_base_url() {
    let c = AnthropicClient::with_base_url(
        "sk-test", "deepseek-chat", "https://api.deepseek.com/anthropic"
    );
    assert_eq!(c.endpoint(), "https://api.deepseek.com/anthropic/v1/messages");
}

#[test]
fn test_client_configuration() {
    let c = AnthropicClient::new("sk-ant-test123", "claude-sonnet-4-20250514")
        .with_max_tokens(4096)
        .with_api_version("2023-06-01");
    assert_eq!(c.model(), "claude-sonnet-4-20250514");
}

// ── Stream event deserialization ────────────────────────────────────────────

#[test]
fn test_stream_event_message_start() {
    let raw = json!({
        "type": "message_start",
        "message": {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude"}
    });
    let ev: StreamEvent = serde_json::from_value(raw).unwrap();
    assert_eq!(ev.event_type, "message_start");
    assert_eq!(ev.message.unwrap().id, "msg_1");
}

#[test]
fn test_stream_event_content_block_start() {
    let raw = json!({
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "tool_use", "id": "tu_1", "name": "run", "input": {}}
    });
    let ev: StreamEvent = serde_json::from_value(raw).unwrap();
    assert_eq!(ev.event_type, "content_block_start");
    let cb = ev.content_block.unwrap();
    match cb {
        ContentBlock::ToolUse { id, name, .. } => {
            assert_eq!(id, "tu_1");
            assert_eq!(name, "run");
        }
        _ => panic!("expected ToolUse"),
    }
}

#[test]
fn test_stream_event_delta() {
    let raw = json!({
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": "Hello"}
    });
    let ev: StreamEvent = serde_json::from_value(raw).unwrap();
    let d = ev.delta.unwrap();
    assert_eq!(d.text.as_deref(), Some("Hello"));
}

// ── Usage ───────────────────────────────────────────────────────────────────

#[test]
fn test_usage_serde() {
    let raw = json!({
        "input_tokens": 100,
        "output_tokens": 50,
        "cache_creation_input_tokens": 10,
        "cache_read_input_tokens": 20
    });
    let usage: Usage = serde_json::from_value(raw).unwrap();
    assert_eq!(usage.input_tokens, 100);
    assert_eq!(usage.output_tokens, 50);
    assert_eq!(usage.cache_creation_input_tokens, Some(10));
    assert_eq!(usage.cache_read_input_tokens, Some(20));
}