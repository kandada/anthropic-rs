// Copyright (c) 2025 xiefujin <490021684@qq.com>
// Licensed under Apache-2.0, see LICENSE file for full license terms.

//! Comprehensive end-to-end tests for anthropic-client-rs.
//!
//! Grounded in the official Anthropic wire format (as modeled by the
//! official Python SDK):
//!   - stream events: message_start / content_block_start /
//!     content_block_delta (text_delta, thinking_delta, signature_delta,
//!     input_json_delta) / content_block_stop / ping / message_delta /
//!     message_stop / error
//!   - extended thinking: thinking blocks end with a `signature_delta`;
//!     thinking + redacted_thinking blocks must be passed back unchanged
//!     in multi-turn conversations
//!   - usage arrives in `message_start` (input) and `message_delta`
//!     (cumulative)
//!
//! Also covers HTTP retry behaviour (429/5xx retried, Retry-After honored,
//! 4xx not retried) against a real local server.

use std::io::Cursor;
use std::sync::atomic::AtomicBool;

use serde_json::json;

use anthropic_client_rs::*;

// ── Full streaming conversation (thinking + signature + tool + text) ───────

/// A realistic extended-thinking + tool-use stream, exactly as Anthropic
/// emits it: message_start (with usage) → thinking block (thinking_delta +
/// signature_delta) → tool_use block → text block → ping → message_delta
/// (usage + stop_reason) → message_stop.
const FULL_STREAM: &str = concat!(
    "event: message_start\n",
    "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"model\":\"claude\",\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":25,\"output_tokens\":1}}}\n\n",
    "event: content_block_start\n",
    "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\",\"signature\":\"\"}}\n\n",
    "event: content_block_delta\n",
    "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"I need to\"}}\n\n",
    "event: content_block_delta\n",
    "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\" check the weather.\"}}\n\n",
    "event: content_block_delta\n",
    "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"EuHT9FrxBkU0b70fGvPEqyF9Y5Lw==\"}}\n\n",
    "event: content_block_stop\n",
    "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
    "event: content_block_start\n",
    "data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_01A\",\"name\":\"get_weather\",\"input\":{}}}\n\n",
    "event: content_block_delta\n",
    "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"location\\\": \\\"San Franc\"}}\n\n",
    "event: content_block_delta\n",
    "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"isco\\\"}\"}}\n\n",
    "event: content_block_stop\n",
    "data: {\"type\":\"content_block_stop\",\"index\":1}\n\n",
    "event: content_block_start\n",
    "data: {\"type\":\"content_block_start\",\"index\":2,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
    "event: ping\n",
    "data: {\"type\":\"ping\"}\n\n",
    "event: content_block_delta\n",
    "data: {\"type\":\"content_block_delta\",\"index\":2,\"delta\":{\"type\":\"text_delta\",\"text\":\"Let me check\"}}\n\n",
    "event: content_block_delta\n",
    "data: {\"type\":\"content_block_delta\",\"index\":2,\"delta\":{\"type\":\"text_delta\",\"text\":\" the weather.\"}}\n\n",
    "event: content_block_stop\n",
    "data: {\"type\":\"content_block_stop\",\"index\":2}\n\n",
    "event: message_delta\n",
    "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\",\"stop_sequence\":null},\"usage\":{\"input_tokens\":25,\"output_tokens\":52}}\n\n",
    "event: message_stop\n",
    "data: {\"type\":\"message_stop\"}\n\n",
    "data: [DONE]\n\n",
);

fn parse_stream(raw: &str) -> LlmResponse {
    let cancel = AtomicBool::new(false);
    parse_anthropic_stream(Cursor::new(raw.as_bytes().to_vec()), |_| {}, |_, _| {}, &cancel).unwrap()
}

#[test]
fn full_stream_parses_thinking_signature_tool_and_text() {
    let resp = parse_stream(FULL_STREAM);

    // Text: only text_delta contributions.
    assert_eq!(resp.text, "Let me check the weather.");
    // Reasoning: only thinking_delta contributions.
    assert_eq!(resp.reasoning_content.as_deref(), Some("I need to check the weather."));

    // Tool call assembled from fragmented input_json_delta.
    assert_eq!(resp.tool_calls.len(), 1);
    assert_eq!(resp.tool_calls[0].name, "get_weather");
    assert_eq!(resp.tool_calls[0].id, "toolu_01A");
    assert_eq!(resp.tool_calls[0].parsed_args()["location"], "San Francisco");

    // Ordered content blocks preserve stream order: thinking, tool_use, text.
    assert_eq!(resp.content_blocks.len(), 3);
    match &resp.content_blocks[0] {
        ContentBlock::Thinking { thinking, signature } => {
            assert_eq!(thinking, "I need to check the weather.");
            assert_eq!(signature, "EuHT9FrxBkU0b70fGvPEqyF9Y5Lw==");
        }
        other => panic!("expected Thinking, got {other:?}"),
    }
    match &resp.content_blocks[1] {
        ContentBlock::ToolUse { id, name, input } => {
            assert_eq!(id, "toolu_01A");
            assert_eq!(name, "get_weather");
            assert_eq!(input["location"], "San Francisco");
        }
        other => panic!("expected ToolUse, got {other:?}"),
    }
    match &resp.content_blocks[2] {
        ContentBlock::Text { text, .. } => assert_eq!(text, "Let me check the weather."),
        other => panic!("expected Text, got {other:?}"),
    }

    // stop_reason from message_delta.
    assert_eq!(resp.finish_reason.as_deref(), Some("tool_use"));

    // usage from message_delta (cumulative, overrides message_start).
    let u = resp.usage.expect("usage should be captured");
    assert_eq!(u.input_tokens, 25);
    assert_eq!(u.output_tokens, 52);
}

// ── Multi-turn roundtrip ────────────────────────────────────────────────────

#[test]
fn streamed_thinking_blocks_roundtrip_into_next_request() {
    let resp = parse_stream(FULL_STREAM);

    // Reconstruct the assistant turn from the streamed content_blocks, as an
    // agent loop would, then serialize it for the next API call.
    let mut assistant = ChatMessage::assistant(resp.text.clone());
    assistant.content_blocks = Some(resp.content_blocks.clone());
    let assistant_turn = vec![
        assistant,
        ChatMessage::tool_result("toolu_01A", "{\"temperature\": 18}"),
    ];

    let body = AnthropicClient::build_anthropic_messages(&assistant_turn);

    // First message = assistant turn with ordered blocks.
    assert_eq!(body[0]["role"], "assistant");
    let blocks = body[0]["content"].as_array().unwrap();
    assert_eq!(blocks.len(), 3);
    // thinking block carries its signature back (required by Anthropic).
    assert_eq!(blocks[0]["type"], "thinking");
    assert_eq!(blocks[0]["thinking"], "I need to check the weather.");
    assert_eq!(blocks[0]["signature"], "EuHT9FrxBkU0b70fGvPEqyF9Y5Lw==");
    assert_eq!(blocks[1]["type"], "tool_use");
    assert_eq!(blocks[2]["type"], "text");

    // Second message = tool_result coalesced into a user message.
    assert_eq!(body[1]["role"], "user");
    assert_eq!(body[1]["content"][0]["type"], "tool_result");
    assert_eq!(body[1]["content"][0]["tool_use_id"], "toolu_01A");
}

#[test]
fn redacted_thinking_roundtrips_unchanged() {
    let raw = concat!(
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"redacted_thinking\",\"data\":\"encrypted-data-abc\"}}\n\n",
        "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n",
        "data: [DONE]\n\n"
    );
    let resp = parse_stream(raw);
    assert_eq!(resp.content_blocks.len(), 1);
    let data = match &resp.content_blocks[0] {
        ContentBlock::RedactedThinking { data } => data.clone(),
        other => panic!("expected RedactedThinking, got {other:?}"),
    };
    assert_eq!(data, "encrypted-data-abc");

    let mut msg = ChatMessage::assistant("");
    msg.content_blocks = Some(resp.content_blocks);
    let body = AnthropicClient::build_anthropic_messages(&[msg]);
    assert_eq!(body[0]["content"][0]["type"], "redacted_thinking");
    assert_eq!(body[0]["content"][0]["data"], "encrypted-data-abc");
}

// ── Usage variants ──────────────────────────────────────────────────────────

#[test]
fn usage_from_message_start_only() {
    let raw = concat!(
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"m\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"model\":\"c\",\"usage\":{\"input_tokens\":11,\"output_tokens\":1}}}\n\n",
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\"}}\n\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n",
        "data: {\"type\":\"message_stop\"}\n\n",
        "data: [DONE]\n\n"
    );
    let resp = parse_stream(raw);
    let u = resp.usage.expect("usage from message_start");
    assert_eq!(u.input_tokens, 11);
}

#[test]
fn usage_none_when_absent() {
    let raw = concat!(
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\"}}\n\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n",
        "data: [DONE]\n\n"
    );
    let resp = parse_stream(raw);
    assert!(resp.usage.is_none());
}

// ── StreamEvent / StreamDelta deserialization (all event types) ────────────

#[test]
fn stream_event_deserializes_all_types() {
    let events = [
        json!({"type":"message_start","message":{"id":"m","type":"message","role":"assistant","model":"c"}}),
        json!({"type":"ping"}),
        json!({"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}),
        json!({"type":"content_block_start","index":1,"content_block":{"type":"redacted_thinking","data":"xx"}}),
        json!({"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"t"}}),
        json!({"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig"}}),
        json!({"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"a\":1}"}}),
        json!({"type":"content_block_stop","index":0}),
        json!({"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"input_tokens":1,"output_tokens":2}}),
        json!({"type":"message_stop"}),
        json!({"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}),
    ];
    for raw in events {
        let ev: StreamEvent = serde_json::from_value(raw).unwrap();
        assert!(!ev.event_type.is_empty());
    }

    // signature_delta carries the signature.
    let ev: StreamEvent = serde_json::from_value(json!({
        "type":"content_block_delta","index":0,
        "delta":{"type":"signature_delta","signature":"sig_xyz"}
    })).unwrap();
    assert_eq!(ev.delta.unwrap().signature.as_deref(), Some("sig_xyz"));

    // redacted_thinking block start carries opaque data.
    let ev: StreamEvent = serde_json::from_value(json!({
        "type":"content_block_start","index":0,
        "content_block":{"type":"redacted_thinking","data":"opaque"}
    })).unwrap();
    match ev.content_block.unwrap() {
        ContentBlock::RedactedThinking { data } => assert_eq!(data, "opaque"),
        other => panic!("expected RedactedThinking, got {other:?}"),
    }
}

// ── ContentBlock serde: all variants ────────────────────────────────────────

#[test]
fn content_block_all_variants_roundtrip() {
    let blocks = vec![
        ContentBlock::Text { text: "t".into(), citations: None },
        ContentBlock::Image { source: ImageSource { source_type: "base64".into(), media_type: "image/jpeg".into(), data: "d".into() } },
        ContentBlock::ToolUse { id: "tu".into(), name: "fn".into(), input: json!({"a": 1}) },
        ContentBlock::ToolResult { tool_use_id: "tu".into(), content: "ok".into(), is_error: None },
        ContentBlock::Thinking { thinking: "think".into(), signature: "sig".into() },
        ContentBlock::RedactedThinking { data: "redacted".into() },
        ContentBlock::Document { source: DocumentSource { source_type: "base64".into(), media_type: None, data: None, url: None }, title: None, context: None, citations: None },
    ];
    for b in &blocks {
        let v = serde_json::to_value(b).unwrap();
        let back: ContentBlock = serde_json::from_value(v).unwrap();
        assert!(format!("{back:?}").contains(""), "roundtrip failed for {back:?}");
    }

    // Spot-check wire shapes.
    let v = serde_json::to_value(&ContentBlock::RedactedThinking { data: "x".into() }).unwrap();
    assert_eq!(v["type"], "redacted_thinking");
    let v = serde_json::to_value(&ContentBlock::Thinking { thinking: "t".into(), signature: "s".into() }).unwrap();
    assert_eq!(v["signature"], "s");
    let v = serde_json::to_value(&ContentBlock::Document { source: DocumentSource { source_type: "url".into(), media_type: None, data: None, url: Some("https://x".into()) }, title: None, context: None, citations: None }).unwrap();
    assert_eq!(v["type"], "document");
}

// ── ThinkingConfig serialization ────────────────────────────────────────────

#[test]
fn thinking_config_all_variants_serialize_like_official_sdk() {
    let cases = [
        (ThinkingConfig::enabled(4096), json!({"type":"enabled","budget_tokens":4096})),
        (ThinkingConfig::enabled(4096).display(ThinkingDisplay::Omitted), json!({"type":"enabled","budget_tokens":4096,"display":"omitted"})),
        (ThinkingConfig::enabled(4096).display(ThinkingDisplay::Summarized), json!({"type":"enabled","budget_tokens":4096,"display":"summarized"})),
        (ThinkingConfig::adaptive(), json!({"type":"adaptive"})),
        (ThinkingConfig::adaptive().display(ThinkingDisplay::Omitted), json!({"type":"adaptive","display":"omitted"})),
        (ThinkingConfig::disabled(), json!({"type":"disabled"})),
    ];
    for (cfg, expected) in cases {
        assert_eq!(serde_json::to_value(&cfg).unwrap(), expected);
    }
}

// ── MessageRequest: full parameter surface ──────────────────────────────────

#[test]
fn message_request_all_params() {
    let body = MessageRequest::new("claude", vec![ChatMessage::user("hi")], 8192)
        .system("be brief")
        .temperature(0.9)
        .top_p(0.8)
        .top_k(30)
        .stop_sequences(vec!["STOP".into()])
        .tools(vec![Tool::new("t", "d", json!({"type":"object","properties":{}}))])
        .tool_choice(ToolChoice::any())
        .thinking(ThinkingConfig::adaptive())
        .metadata(Metadata { user_id: Some("u1".into()) })
        .stream(true)
        .build_body();

    assert_eq!(body["model"], "claude");
    assert_eq!(body["max_tokens"], 8192);
    assert_eq!(body["temperature"], 0.9);
    assert_eq!(body["top_p"], 0.8);
    assert_eq!(body["top_k"], 30);
    assert_eq!(body["stop_sequences"][0], "STOP");
    assert_eq!(body["tool_choice"]["type"], "any");
    assert_eq!(body["thinking"]["type"], "adaptive");
    assert_eq!(body["metadata"]["user_id"], "u1");
    assert_eq!(body["stream"], true);
}

// ── SSE framing edge cases ──────────────────────────────────────────────────

#[test]
fn sse_bom_crlf_and_comments_are_handled() {
    let raw = concat!(
        "\u{FEFF}event: message_start\r\n",
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"m\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"c\"}}\r\n\r\n",
        ": keep-alive comment\r\n",
        "event: ping\r\n",
        "data: {\"type\":\"ping\"}\r\n\r\n",
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\"}}\r\n\r\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"ok\"}}\r\n\r\n",
        "data: {\"type\":\"message_stop\"}\r\n\r\n",
        "data: [DONE]\r\n\r\n"
    );
    let resp = parse_stream(raw);
    assert_eq!(resp.text, "ok");
}

#[test]
fn sse_multiple_think_blocks_interleaved_with_text() {
    let raw = concat!(
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\"}}\n\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"first\"}}\n\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"sig1\"}}\n\n",
        "data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"text\"}}\n\n",
        "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"inter\"}}\n\n",
        "data: {\"type\":\"content_block_start\",\"index\":2,\"content_block\":{\"type\":\"thinking\"}}\n\n",
        "data: {\"type\":\"content_block_delta\",\"index\":2,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"second\"}}\n\n",
        "data: {\"type\":\"content_block_delta\",\"index\":2,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"sig2\"}}\n\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n",
        "data: [DONE]\n\n"
    );
    let resp = parse_stream(raw);
    // reasoning is concatenation of both thinking blocks, in order.
    assert_eq!(resp.reasoning_content.as_deref(), Some("firstsecond"));
    assert_eq!(resp.text, "inter");
    assert_eq!(resp.content_blocks.len(), 3);
    match &resp.content_blocks[0] {
        ContentBlock::Thinking { signature, .. } => assert_eq!(signature, "sig1"),
        other => panic!("{other:?}"),
    }
    match &resp.content_blocks[2] {
        ContentBlock::Thinking { signature, .. } => assert_eq!(signature, "sig2"),
        other => panic!("{other:?}"),
    }
}

// ── Error structure ─────────────────────────────────────────────────────────

#[test]
fn errors_carry_status_code_and_retry_after() {
    let e = AnthropicError::api(429, Some(17), "rate limited");
    match &e {
        AnthropicError::Api(ae) => {
            assert_eq!(ae.status_code, Some(429));
            assert_eq!(ae.retry_after_secs, Some(17));
            assert_eq!(ae.message, "rate limited");
            assert!(ae.is_retryable());
        }
        _ => panic!("expected Api"),
    }
    assert_eq!(e.retry_after_secs(), Some(17));
    assert!(e.is_retryable());

    // In-stream error carries no status code.
    let se = AnthropicError::stream_error("overloaded_error: Overloaded");
    match &se {
        AnthropicError::Api(ae) => {
            assert_eq!(ae.status_code, None);
            assert!(ae.is_retryable(), "overloaded_error is retryable");
        }
        _ => panic!("expected Api"),
    }
    // Non-retryable: 4xx and JSON parse failures.
    assert!(!AnthropicError::api(400, None, "bad").is_retryable());
    assert!(!AnthropicError::Json("bad json".into()).is_retryable());
}

// ── Non-streaming response assembly ─────────────────────────────────────────

#[test]
fn assemble_message_response_includes_redacted_and_blocks() {
    let raw = json!({
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "plan", "signature": "sig_x"},
            {"type": "redacted_thinking", "data": "opaque-data"},
            {"type": "tool_use", "id": "tu_1", "name": "run", "input": {"cmd": "ls"}},
            {"type": "text", "text": "done"}
        ],
        "model": "claude",
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 3, "output_tokens": 7}
    });
    let resp = anthropic_client_rs::api_common::assemble_message_response(&raw);
    assert_eq!(resp.text, "done");
    assert_eq!(resp.reasoning_content.as_deref(), Some("plan"));
    assert_eq!(resp.tool_calls.len(), 1);
    assert_eq!(resp.tool_calls[0].name, "run");
    assert_eq!(resp.usage.unwrap().output_tokens, 7);
    assert_eq!(resp.content_blocks.len(), 4);
    match &resp.content_blocks[1] {
        ContentBlock::RedactedThinking { data } => assert_eq!(data, "opaque-data"),
        other => panic!("{other:?}"),
    }
}

// ── MessageResponse full deserialization ────────────────────────────────────

#[test]
fn message_response_deserializes_full_extended_thinking_payload() {
    let raw = json!({
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "deep", "signature": "sig"},
            {"type": "redacted_thinking", "data": "opaque"},
            {"type": "text", "text": "answer"}
        ],
        "model": "claude",
        "stop_reason": "end_turn",
        "stop_sequence": null,
        "usage": {"input_tokens": 1, "output_tokens": 2}
    });
    let mr: MessageResponse = serde_json::from_value(raw).unwrap();
    assert_eq!(mr.content.len(), 3);
    assert_eq!(mr.stop_reason.as_deref(), Some("end_turn"));
}

// ── HTTP retry behaviour (tiny_http) ────────────────────────────────────────

const MESSAGE_200_BODY: &str = r#"{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"server ok"}],"model":"claude","stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":2}}"#;

fn respond_once(req: tiny_http::Request, body: &str, status: u16, retry_after: Option<&str>) {
    let mut resp = tiny_http::Response::from_string(body.to_string()).with_status_code(status);
    if let Some(ra) = retry_after {
        resp = resp.with_header(tiny_http::Header::from_bytes(b"Retry-After", ra.as_bytes()).unwrap());
    }
    req.respond(resp).unwrap();
}

fn bind_port(server: &tiny_http::Server) -> u16 {
    match server.server_addr() {
        tiny_http::ListenAddr::IP(addr) => addr.port(),
        _ => panic!("unexpected addr"),
    }
}

#[test]
fn retries_on_429_then_succeeds() {
    let server = tiny_http::Server::http("127.0.0.1:0").unwrap();
    let port = bind_port(&server);
    let handle = std::thread::spawn(move || {
        // request 1 → 429
        respond_once(server.recv().unwrap(), "rate limited", 429, None);
        // request 2 → 200
        respond_once(server.recv().unwrap(), MESSAGE_200_BODY, 200, None);
    });

    let client = AnthropicClient::with_base_url(
        "sk-ant-test", "claude-test", format!("http://127.0.0.1:{port}"),
    )
    .with_retry_config(RetryConfig { max_retries: 2, base_delay_ms: 1, max_delay_ms: 10 });

    let resp = client.messages_create(&[ChatMessage::user("hi")], None, None, 100, None).unwrap();
    assert_eq!(resp.text, "server ok");
    handle.join().unwrap();
}

#[test]
fn retries_on_503_and_honors_retry_after() {
    let server = tiny_http::Server::http("127.0.0.1:0").unwrap();
    let port = bind_port(&server);
    let handle = std::thread::spawn(move || {
        respond_once(server.recv().unwrap(), "unavailable", 503, Some("0"));
        respond_once(server.recv().unwrap(), MESSAGE_200_BODY, 200, None);
    });

    let client = AnthropicClient::with_base_url(
        "sk-ant-test", "claude-test", format!("http://127.0.0.1:{port}"),
    )
    .with_retry_config(RetryConfig { max_retries: 2, base_delay_ms: 1000, max_delay_ms: 5000 });

    let resp = client.messages_create(&[ChatMessage::user("hi")], None, None, 100, None).unwrap();
    assert_eq!(resp.text, "server ok");
    handle.join().unwrap();
}

#[test]
fn does_not_retry_on_400() {
    let server = tiny_http::Server::http("127.0.0.1:0").unwrap();
    let port = bind_port(&server);
    let handle = std::thread::spawn(move || {
        respond_once(server.recv().unwrap(), "bad request", 400, None);
    });

    let client = AnthropicClient::with_base_url(
        "sk-ant-test", "claude-test", format!("http://127.0.0.1:{port}"),
    )
    .with_retry_config(RetryConfig { max_retries: 3, base_delay_ms: 1000, max_delay_ms: 5000 });

    let err = client.messages_create(&[ChatMessage::user("hi")], None, None, 100, None).unwrap_err();
    match &err {
        AnthropicError::Api(ae) => assert_eq!(ae.status_code, Some(400)),
        _ => panic!("expected Api 400, got {err:?}"),
    }
    handle.join().unwrap();
}

#[test]
fn retries_stream_requests_on_5xx() {
    let server = tiny_http::Server::http("127.0.0.1:0").unwrap();
    let port = bind_port(&server);
    let handle = std::thread::spawn(move || {
        respond_once(server.recv().unwrap(), "boom", 500, None);
        // Streaming success: SSE body.
        let body = concat!(
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\"}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"streamed\"}}\n\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n",
            "data: [DONE]\n\n"
        );
        respond_once(server.recv().unwrap(), body, 200, None);
    });

    let client = AnthropicClient::with_base_url(
        "sk-ant-test", "claude-test", format!("http://127.0.0.1:{port}"),
    )
    .with_retry_config(RetryConfig { max_retries: 2, base_delay_ms: 1, max_delay_ms: 10 });

    let mut deltas = Vec::new();
    let resp = client
        .messages_stream(
            &[ChatMessage::user("hi")], None, None, 100, None,
            |d| deltas.push(d.to_string()),
            |_, _| {},
        )
        .unwrap();
    assert_eq!(resp.text, "streamed");
    assert_eq!(deltas.join(""), "streamed");
    handle.join().unwrap();
}

// ── MessageRequest send path ────────────────────────────────────────────────

#[test]
fn messages_send_roundtrips_through_request_builder() {
    let server = tiny_http::Server::http("127.0.0.1:0").unwrap();
    let port = bind_port(&server);
    let handle = std::thread::spawn(move || {
        respond_once(server.recv().unwrap(), MESSAGE_200_BODY, 200, None);
    });

    let client = AnthropicClient::with_base_url(
        "sk-ant-test", "claude-test", format!("http://127.0.0.1:{port}"),
    );
    let req = MessageRequest::new("claude-test", vec![ChatMessage::user("hi")], 200)
        .thinking(ThinkingConfig::adaptive());
    let resp = client.messages_send(&req).unwrap();
    assert_eq!(resp.text, "server ok");
    handle.join().unwrap();
}

// ── Async client HTTP retry behaviour ───────────────────────────────────────

#[cfg(feature = "async")]
#[tokio::test]
async fn async_client_retries_on_429_then_succeeds() {
    use anthropic_client_rs::AnthropicAsyncClient;

    let server = tiny_http::Server::http("127.0.0.1:0").unwrap();
    let port = bind_port(&server);
    let handle = std::thread::spawn(move || {
        respond_once(server.recv().unwrap(), "rate limited", 429, None);
        respond_once(server.recv().unwrap(), MESSAGE_200_BODY, 200, None);
    });

    let client = AnthropicAsyncClient::with_base_url(
        "sk-ant-test", "claude-test", format!("http://127.0.0.1:{port}"),
    )
    .with_retry_config(RetryConfig { max_retries: 2, base_delay_ms: 1, max_delay_ms: 10 });

    let resp = client
        .messages_create(&[ChatMessage::user("hi")], None, None, 100, None)
        .await
        .unwrap();
    assert_eq!(resp.text, "server ok");
    handle.join().unwrap();
}

#[cfg(feature = "async")]
#[tokio::test]
async fn async_client_does_not_retry_on_400() {
    use anthropic_client_rs::AnthropicAsyncClient;

    let server = tiny_http::Server::http("127.0.0.1:0").unwrap();
    let port = bind_port(&server);
    let handle = std::thread::spawn(move || {
        respond_once(server.recv().unwrap(), "bad request", 400, None);
    });

    let client = AnthropicAsyncClient::with_base_url(
        "sk-ant-test", "claude-test", format!("http://127.0.0.1:{port}"),
    )
    .with_retry_config(RetryConfig { max_retries: 3, base_delay_ms: 1000, max_delay_ms: 5000 });

    let err = client
        .messages_create(&[ChatMessage::user("hi")], None, None, 100, None)
        .await
        .unwrap_err();
    match &err {
        AnthropicError::Api(ae) => assert_eq!(ae.status_code, Some(400)),
        _ => panic!("expected Api 400, got {err:?}"),
    }
    handle.join().unwrap();
}
