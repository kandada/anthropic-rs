// Copyright (c) 2025 xiefujin <490021684@qq.com>
// Licensed under Apache-2.0, see LICENSE file for full license terms.

//! Edge-case tests for anthropic-rs.
//!
//! Covers wire-format corner cases and failure paths that happy-path tests
//! miss: empty/keep-alive streams, out-of-order blocks, malformed tool
//! input, unicode content, retry classification matrix, and HTTP retry
//! edge behaviour (408, exhaustion, max_retries=0).

use std::io::Cursor;
use std::sync::atomic::AtomicBool;

use serde_json::json;

use anthropic_client_rs::*;

fn parse_stream(raw: &str) -> LlmResponse {
    let cancel = AtomicBool::new(false);
    parse_anthropic_stream(Cursor::new(raw.as_bytes().to_vec()), |_| {}, |_, _| {}, &cancel).unwrap()
}

// ── Empty / keep-alive streams ─────────────────────────────────────────────

#[test]
fn empty_stream_is_ok_with_empty_response() {
    let resp = parse_stream("data: [DONE]\n\n");
    assert!(resp.text.is_empty());
    assert!(resp.tool_calls.is_empty());
    assert!(resp.content_blocks.is_empty());
    assert!(resp.finish_reason.is_none());
    assert!(resp.usage.is_none());
}

#[test]
fn keep_alive_only_stream_is_ok() {
    // Anthropic servers send periodic `ping` events; the stream may contain
    // nothing else.
    let raw = concat!(
        ": keep-alive comment\n\n",
        "data: {\"type\":\"ping\"}\n\n",
        "data: {\"type\":\"ping\"}\n\n",
        "data: [DONE]\n\n"
    );
    let resp = parse_stream(raw);
    assert!(resp.text.is_empty());
    assert!(resp.reasoning_content.is_none());
}

// ── Out-of-order / duplicate blocks ────────────────────────────────────────

#[test]
fn delta_before_content_block_start_is_tolerated() {
    let raw = concat!(
        "data: {\"type\":\"content_block_delta\",\"index\":7,\"delta\":{\"type\":\"text_delta\",\"text\":\"hello\"}}\n\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n",
        "data: [DONE]\n\n"
    );
    let resp = parse_stream(raw);
    assert_eq!(resp.text, "hello");
}

#[test]
fn duplicate_content_block_start_last_wins() {
    let raw = concat!(
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\"}}\n\n",
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"t1\",\"name\":\"run\"}}\n\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"a\\\":1}\"}}\n\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"}}\n\n",
        "data: [DONE]\n\n"
    );
    let resp = parse_stream(raw);
    assert_eq!(resp.tool_calls.len(), 1);
    assert_eq!(resp.tool_calls[0].name, "run");
}

#[test]
fn content_block_stop_for_unknown_index_ignored() {
    let raw = concat!(
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\"}}\n\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"ok\"}}\n\n",
        "data: {\"type\":\"content_block_stop\",\"index\":99}\n\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n",
        "data: [DONE]\n\n"
    );
    let resp = parse_stream(raw);
    assert_eq!(resp.text, "ok");
}

// ── Tool input edge cases ──────────────────────────────────────────────────

#[test]
fn tool_use_without_input_json_defaults_to_empty_object() {
    let raw = concat!(
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"t1\",\"name\":\"run\"}}\n\n",
        "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"}}\n\n",
        "data: [DONE]\n\n"
    );
    let resp = parse_stream(raw);
    assert_eq!(resp.tool_calls.len(), 1);
    assert_eq!(resp.tool_calls[0].arguments, "{}");
    assert_eq!(resp.tool_calls[0].parsed_args(), json!({}));
}

#[test]
fn tool_use_with_invalid_partial_json_falls_back_to_empty_object() {
    let raw = concat!(
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"t1\",\"name\":\"run\"}}\n\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{not json\"}}\n\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"}}\n\n",
        "data: [DONE]\n\n"
    );
    let resp = parse_stream(raw);
    assert_eq!(resp.tool_calls[0].arguments, "{not json");
    assert_eq!(resp.tool_calls[0].parsed_args(), json!({}));
}

// ── Content edge cases ─────────────────────────────────────────────────────

#[test]
fn multiple_text_blocks_concatenate_in_index_order() {
    let raw = concat!(
        "data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"text\"}}\n\n",
        "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"B\"}}\n\n",
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\"}}\n\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"A\"}}\n\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n",
        "data: [DONE]\n\n"
    );
    let resp = parse_stream(raw);
    // text is accumulated in arrival order (streaming), blocks are re-ordered.
    assert_eq!(resp.text, "BA");
    assert_eq!(resp.content_blocks.len(), 2);
    match &resp.content_blocks[0] {
        ContentBlock::Text { text, .. } => assert_eq!(text, "A"),
        other => panic!("{other:?}"),
    }
}

#[test]
fn unicode_content_preserved() {
    let raw = concat!(
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\"}}\n\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"你好，世界 🚀\"}}\n\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n",
        "data: [DONE]\n\n"
    );
    let resp = parse_stream(raw);
    assert_eq!(resp.text, "你好，世界 🚀");
}

#[test]
fn thinking_block_with_empty_content_is_skipped() {
    let raw = concat!(
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\"}}\n\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"sig\"}}\n\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n",
        "data: [DONE]\n\n"
    );
    let resp = parse_stream(raw);
    assert!(resp.reasoning_content.is_none());
    assert!(resp.content_blocks.is_empty());
}

// ── Non-streaming assembly edge cases ──────────────────────────────────────

#[test]
fn assemble_empty_content() {
    let raw = json!({"id":"m","type":"message","role":"assistant","content":[],"model":"c"});
    let resp = anthropic_client_rs::api_common::assemble_message_response(&raw);
    assert!(resp.text.is_empty());
    assert!(resp.tool_calls.is_empty());
    assert!(resp.content_blocks.is_empty());
}

#[test]
fn assemble_redacted_only() {
    let raw = json!({
        "id":"m","type":"message","role":"assistant","model":"c",
        "content":[{"type":"redacted_thinking","data":"opaque"}],
        "stop_reason":"end_turn"
    });
    let resp = anthropic_client_rs::api_common::assemble_message_response(&raw);
    assert!(resp.reasoning_content.is_none());
    assert_eq!(resp.content_blocks.len(), 1);
    match &resp.content_blocks[0] {
        ContentBlock::RedactedThinking { data } => assert_eq!(data, "opaque"),
        other => panic!("{other:?}"),
    }
}

// ── Tool / message shape edge cases ────────────────────────────────────────

#[test]
fn tool_result_is_error_flag_serde() {
    let block = ContentBlock::ToolResult {
        tool_use_id: "t1".into(),
        content: "boom".into(),
        is_error: Some(true),
    };
    let v = serde_json::to_value(&block).unwrap();
    assert_eq!(v["is_error"], true);
}

#[test]
fn tool_choice_all_variants_serde() {
    let auto = serde_json::to_value(ToolChoice::auto()).unwrap();
    assert_eq!(auto["type"], "auto");
    let any = serde_json::to_value(ToolChoice::any()).unwrap();
    assert_eq!(any["type"], "any");
    let specific = serde_json::to_value(ToolChoice::specific("fn")).unwrap();
    assert_eq!(specific["type"], "tool");
    assert_eq!(specific["name"], "fn");
}

#[test]
fn assistant_content_blocks_take_precedence_over_content() {
    let mut msg = ChatMessage::assistant("plain text");
    msg.content_blocks = Some(vec![ContentBlock::Thinking {
        thinking: "think".into(),
        signature: "sig".into(),
    }]);
    let out = AnthropicClient::build_anthropic_messages(&[msg]);
    let blocks = out[0]["content"].as_array().unwrap();
    assert_eq!(blocks.len(), 1);
    assert_eq!(blocks[0]["type"], "thinking");
    assert_eq!(blocks[0]["signature"], "sig");
}

#[test]
fn user_with_blocks_roundtrips() {
    let msg = ChatMessage::user_with_blocks(vec![
        ContentBlock::Text { text: "look".into(), citations: None },
        ContentBlock::ToolResult { tool_use_id: "t1".into(), content: "result".into(), is_error: None },
    ]);
    let out = AnthropicClient::build_anthropic_messages(&[msg]);
    let content = out[0]["content"].as_array().unwrap();
    assert_eq!(content.len(), 2);
    assert_eq!(content[1]["type"], "tool_result");
    assert_eq!(content[1]["tool_use_id"], "t1");
}

// ── Retry classification matrix ─────────────────────────────────────────────

#[test]
fn usage_output_tokens_details_and_server_tool_use_deserialize() {
    let raw = json!({
        "input_tokens": 100,
        "output_tokens": 60,
        "cache_creation_input_tokens": 10,
        "cache_read_input_tokens": 20,
        "output_tokens_details": {
            "reasoning_tokens": 40,
            "accepted_prediction_tokens": 5,
            "rejected_prediction_tokens": 2
        },
        "server_tool_use": {
            "tool_use_count": 3,
            "input_tokens": 30,
            "cache_creation_input_tokens": 1,
            "cache_read_input_tokens": 2
        }
    });
    let u: Usage = serde_json::from_value(raw).unwrap();
    assert_eq!(u.input_tokens, 100);
    let d = u.output_tokens_details.unwrap();
    assert_eq!(d.reasoning_tokens, Some(40));
    let s = u.server_tool_use.unwrap();
    assert_eq!(s.tool_use_count, Some(3));
}

#[test]
fn server_tool_use_content_block_serde() {
    let block = ContentBlock::ServerToolUse {
        id: "st_1".into(),
        name: "bash".into(),
        input: json!({"command": "ls"}),
    };
    let v = serde_json::to_value(&block).unwrap();
    assert_eq!(v["type"], "server_tool_use");
    let back: ContentBlock = serde_json::from_value(v).unwrap();
    match back {
        ContentBlock::ServerToolUse { name, .. } => assert_eq!(name, "bash"),
        other => panic!("{other:?}"),
    }
}

#[test]
fn stream_delta_container_deserialize() {
    let raw = json!({
        "type": "content_block_delta",
        "index": 0,
        "delta": {
            "type": "text_delta",
            "text": "running",
            "container": {"id": "ctr_1", "tool_name": "bash"}
        }
    });
    let ev: StreamEvent = serde_json::from_value(raw).unwrap();
    let d = ev.delta.unwrap();
    assert_eq!(d.container.as_ref().unwrap()["tool_name"], "bash");
}

#[test]
fn count_tokens_hits_count_endpoint() {
    let server = tiny_http::Server::http("127.0.0.1:0").unwrap();
    let port = port_of(&server);
    let handle = std::thread::spawn(move || {
        respond(server.recv().unwrap(), r#"{"input_tokens": 42}"#, 200, None);
    });
    let client = AnthropicClient::with_base_url("sk-ant-test", "c", format!("http://127.0.0.1:{port}"));
    let resp = client
        .count_tokens(&[ChatMessage::user("hi")], None, None)
        .unwrap();
    assert_eq!(resp.input_tokens, 42);
    handle.join().unwrap();
}

#[test]
fn retryable_status_code_matrix() {
    for code in [408u16, 409, 429, 500, 502, 503, 504] {
        assert!(AnthropicError::api(code, None, "x").is_retryable(), "code {code}");
    }
    for code in [400u16, 401, 403, 404, 405, 410, 422, 451] {
        assert!(!AnthropicError::api(code, None, "x").is_retryable(), "code {code}");
    }
    assert!(AnthropicError::Network("timeout".into()).is_retryable());
    assert!(!AnthropicError::Json("bad".into()).is_retryable());
    assert!(!AnthropicError::Io("read failed".into()).is_retryable());
    assert!(!AnthropicError::Cancelled.is_retryable());
}

// ── HTTP retry edge behaviour (tiny_http) ──────────────────────────────────

const OK_BODY: &str = r#"{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"ok"}],"model":"c","stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":2}}"#;

fn respond(req: tiny_http::Request, body: &str, status: u16, retry_after: Option<&str>) {
    let mut resp = tiny_http::Response::from_string(body.to_string()).with_status_code(status);
    if let Some(ra) = retry_after {
        resp = resp.with_header(tiny_http::Header::from_bytes(b"Retry-After", ra.as_bytes()).unwrap());
    }
    req.respond(resp).unwrap();
}

fn port_of(server: &tiny_http::Server) -> u16 {
    match server.server_addr() {
        tiny_http::ListenAddr::IP(a) => a.port(),
        _ => panic!(),
    }
}

fn client(base: String, retries: u32) -> AnthropicClient {
    AnthropicClient::with_base_url("sk-ant-test", "c", base).with_retry_config(RetryConfig {
        max_retries: retries,
        base_delay_ms: 1,
        max_delay_ms: 10,
    })
}

#[test]
fn retries_on_408_request_timeout() {
    let server = tiny_http::Server::http("127.0.0.1:0").unwrap();
    let port = port_of(&server);
    let handle = std::thread::spawn(move || {
        respond(server.recv().unwrap(), "timeout", 408, None);
        respond(server.recv().unwrap(), OK_BODY, 200, None);
    });
    let resp = client(format!("http://127.0.0.1:{port}"), 2)
        .messages_create(&[ChatMessage::user("hi")], None, None, 100, None)
        .unwrap();
    assert_eq!(resp.text, "ok");
    handle.join().unwrap();
}

#[test]
fn retry_exhaustion_returns_last_error() {
    let server = tiny_http::Server::http("127.0.0.1:0").unwrap();
    let port = port_of(&server);
    let handle = std::thread::spawn(move || {
        // max_retries=1 → first attempt + 1 retry
        respond(server.recv().unwrap(), "unavailable", 503, None);
        respond(server.recv().unwrap(), "unavailable", 503, None);
    });
    let err = client(format!("http://127.0.0.1:{port}"), 1)
        .messages_create(&[ChatMessage::user("hi")], None, None, 100, None)
        .unwrap_err();
    match &err {
        AnthropicError::Api(ae) => assert_eq!(ae.status_code, Some(503)),
        other => panic!("expected Api 503, got {other:?}"),
    }
    handle.join().unwrap();
}

#[test]
fn max_retries_zero_never_retries() {
    let server = tiny_http::Server::http("127.0.0.1:0").unwrap();
    let port = port_of(&server);
    let handle = std::thread::spawn(move || {
        respond(server.recv().unwrap(), "unavailable", 503, None);
    });
    let err = client(format!("http://127.0.0.1:{port}"), 0)
        .messages_create(&[ChatMessage::user("hi")], None, None, 100, None)
        .unwrap_err();
    match &err {
        AnthropicError::Api(ae) => assert_eq!(ae.status_code, Some(503)),
        other => panic!("expected Api 503, got {other:?}"),
    }
    handle.join().unwrap();
}
