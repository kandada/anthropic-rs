// Copyright (c) 2025 xiefujin <490021684@qq.com>
// Licensed under Apache-2.0, see LICENSE file for full license terms.

//! End-to-end tests against real APIs.
//!
//! Run with:
//!   MINIMAX=1 cargo test --test e2e -- --nocapture
//!   ALL=1 cargo test --test e2e -- --nocapture

use serde_json::json;
use std::env;

fn minimax_key() -> String {
    env::var("MINIMAX_KEY").unwrap_or_else(|_| {
        "sk-api-guahMZ4fq1miYrkuXkC8Oash352SZlHuB1tXha0bVgHD48w5vM9oe_UYNjK5Mf8aDGOSBNO95Vq0NdKxF0oOiM__PyK8lGG17AFGsLmtvjUetX-JC1L7PrQ".into()
    })
}
fn minimax_url() -> String {
    env::var("MINIMAX_URL").unwrap_or_else(|_| "https://api.minimax.chat/anthropic".into())
}
fn minimax_model() -> String {
    env::var("MINIMAX_MODEL").unwrap_or_else(|_| "MiniMax-M2.7".into())
}

fn should_run(provider: &str) -> bool {
    env::var("ALL").is_ok() || env::var(provider).is_ok()
}

// ── MiniMax tests ──────────────────────────────────────────────────────────

#[test]
fn minimax_basic_message() {
    if !should_run("MINIMAX") { return; }
    let client = anthropic_rs::AnthropicClient::with_base_url(
        minimax_key(), minimax_model(), minimax_url(),
    );
    let resp = client.messages_create(
        &[anthropic_rs::ChatMessage::user("Say one word: hello")],
        None, None, 256,
    ).expect("messages_create failed");
    println!("[minimax] message: '{}'", resp.text.trim());
    assert!(!resp.text.is_empty(), "response should not be empty");
}

#[test]
fn minimax_streaming() {
    if !should_run("MINIMAX") { return; }
    let client = anthropic_rs::AnthropicClient::with_base_url(
        minimax_key(), minimax_model(), minimax_url(),
    );
    let mut deltas = Vec::new();
    client.messages_stream(
        &[anthropic_rs::ChatMessage::user("Count from 1 to 3, one per line.")],
        None, None, 512,
        |d| { deltas.push(d.to_string()); },
        |_, _| {},
    ).expect("messages_stream failed");
    let text = deltas.join("");
    println!("[minimax] stream: {}", text.trim());
    assert!(!text.is_empty());
    assert!(text.contains("1") && text.contains("2") && text.contains("3"));
}

#[test]
fn minimax_tool_use() {
    if !should_run("MINIMAX") { return; }
    let client = anthropic_rs::AnthropicClient::with_base_url(
        minimax_key(), minimax_model(), minimax_url(),
    );
    let tools = vec![anthropic_rs::Tool::new(
        "get_weather",
        "Get weather for a location",
        json!({
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"}
            },
            "required": ["location"]
        }),
    )];
    let resp = client.messages_create(
        &[anthropic_rs::ChatMessage::user("Weather in Tokyo?")],
        None, Some(&tools), 512,
    ).expect("tool use failed");
    println!("[minimax] tools: {:?}", resp.tool_calls.iter().map(|t| &t.name).collect::<Vec<_>>());
    assert!(!resp.tool_calls.is_empty() || !resp.text.is_empty());
    if !resp.tool_calls.is_empty() {
        let tc = &resp.tool_calls[0];
        assert_eq!(tc.name, "get_weather");
        println!("[minimax] tool call: {} -> {}", tc.name, tc.arguments);
    }
}

#[test]
fn minimax_validate_api_key() {
    if !should_run("MINIMAX") { return; }
    let client = anthropic_rs::AnthropicClient::with_base_url(
        minimax_key(), minimax_model(), minimax_url(),
    );
    client.validate().expect("validate should succeed");
    println!("[minimax] validate: OK");
}

#[test]
fn minimax_invalid_key() {
    let client = anthropic_rs::AnthropicClient::with_base_url(
        "sk-invalid-key-12345", minimax_model(), minimax_url(),
    );
    let err = client.validate().unwrap_err();
    println!("[minimax] invalid key error: {err}");
    assert!(!err.is_retryable(), "auth error should not be retryable");
}

#[test]
fn minimax_system_prompt() {
    if !should_run("MINIMAX") { return; }
    let client = anthropic_rs::AnthropicClient::with_base_url(
        minimax_key(), minimax_model(), minimax_url(),
    );
    let resp = client.messages_create(
        &[anthropic_rs::ChatMessage::user("Say one word: hello")],
        Some("Reply with exactly one word."),
        None, 128,
    ).expect("system prompt failed");
    println!("[minimax] system: '{}' (stop: {:?})", resp.text.trim(), resp.finish_reason);
    assert!(!resp.text.is_empty(), "empty response, finish_reason={:?}", resp.finish_reason);
}

#[test]
fn minimax_temperature() {
    if !should_run("MINIMAX") { return; }
    let client = anthropic_rs::AnthropicClient::with_base_url(
        minimax_key(), minimax_model(), minimax_url(),
    );
    let req = anthropic_rs::MessageRequest::new(
        minimax_model(),
        vec![anthropic_rs::ChatMessage::user("Say 'OK'.")],
        128,
    ).temperature(0.0);
    let body = req.build_body();
    let resp = client.post_json(body).expect("request failed");
    let raw: serde_json::Value = resp.into_json().expect("json parse failed");
    // Use the shared assembler to extract text from content blocks
    let llm_resp = anthropic_rs::api_common::assemble_message_response(&raw);
    let text = &llm_resp.text;
    println!("[minimax] temperature: '{text}' (stop: {:?})", llm_resp.finish_reason);
    assert!(!text.is_empty(), "response text empty. Raw: {raw}");
}

#[test]
fn minimax_request_builder() {
    if !should_run("MINIMAX") { return; }
    let client = anthropic_rs::AnthropicClient::with_base_url(
        minimax_key(), minimax_model(), minimax_url(),
    );
    let req = anthropic_rs::MessageRequest::new(
        minimax_model(),
        vec![anthropic_rs::ChatMessage::user("Say 'hello'.")],
        128,
    ).temperature(0.3).top_p(0.9);
    let body = req.build_body();
    assert_eq!(body["temperature"], 0.3);
    assert_eq!(body["top_p"], 0.9);
    let resp = client.post_json(body).expect("request failed");
    let raw: serde_json::Value = resp.into_json().expect("json parse failed");
    let llm_resp = anthropic_rs::api_common::assemble_message_response(&raw);
    println!("[minimax] builder: '{}'", llm_resp.text.trim());
    assert!(!llm_resp.text.is_empty(), "empty from builder: {raw}");
}

// ── Multi-turn tool roundtrip ──────────────────────────────────────────────

#[test]
fn minimax_multi_turn_tool_roundtrip() {
    if !should_run("MINIMAX") { return; }
    let client = anthropic_rs::AnthropicClient::with_base_url(
        minimax_key(), minimax_model(), minimax_url(),
    );
    let tools = vec![anthropic_rs::Tool::new(
        "get_weather",
        "Get weather for a location",
        json!({
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"]
        }),
    )];
    let resp1 = client.messages_create(
        &[anthropic_rs::ChatMessage::user("Weather in Tokyo? Use the tool.")],
        None, Some(&tools), 512,
    ).expect("turn 1 failed");
    println!("[minimax multi] turn1: {:?}", resp1.tool_calls.iter().map(|t| (&t.name, &t.arguments)).collect::<Vec<_>>());
    assert!(!resp1.tool_calls.is_empty(), "expected tool call, got: {}", resp1.text);
    let tc = &resp1.tool_calls[0];

    let messages = vec![
        anthropic_rs::ChatMessage::user("Weather in Tokyo? Use the tool."),
        anthropic_rs::ChatMessage::assistant_with_tools("", vec![anthropic_rs::ToolCall {
            id: tc.id.clone(), call_type: "tool_use".into(),
            name: tc.name.clone(), input: serde_json::from_str(&tc.arguments).unwrap_or_default(),
        }]),
        anthropic_rs::ChatMessage::tool_result(&tc.id, "Sunny, 25C"),
        anthropic_rs::ChatMessage::user("Summarize in one sentence."),
    ];
    let resp2 = client.messages_create(&messages, None, None, 512).expect("turn 2 failed");
    println!("[minimax multi] turn2: '{}'", resp2.text.trim());
    assert!(!resp2.text.is_empty());
    assert!(resp2.text.to_lowercase().contains("sunny") || resp2.text.contains("25"));
}

// ── Streaming vs non-streaming ─────────────────────────────────────────────

#[test]
fn minimax_streaming_vs_non_streaming() {
    if !should_run("MINIMAX") { return; }
    let client = anthropic_rs::AnthropicClient::with_base_url(
        minimax_key(), minimax_model(), minimax_url(),
    );
    let msg = anthropic_rs::ChatMessage::user("Count 1,2,3,4,5 comma separated, no other text.");

    let non_stream = client.messages_create(&[msg.clone()], None, None, 256).expect("non-stream failed");
    println!("[minimax cmp] non-stream: '{}' (len={})", non_stream.text.trim(), non_stream.text.len());

    let mut deltas = Vec::new();
    client.messages_stream(&[msg], None, None, 256, |d| { deltas.push(d.to_string()); }, |_, _| {}).expect("stream failed");
    let stream_text = deltas.join("");
    println!("[minimax cmp] stream: '{}' (len={})", stream_text.trim(), stream_text.len());

    assert!(!non_stream.text.is_empty());
    assert!(!stream_text.is_empty());
    println!("[minimax cmp] OK: non_stream={} chars, stream={} chars", non_stream.text.len(), stream_text.len());
}

// ── Real multimodal ────────────────────────────────────────────────────────

#[test]
fn minimax_real_image_recognition() {
    if !should_run("MINIMAX") { return; }
    let client = anthropic_rs::AnthropicClient::with_base_url(
        minimax_key(), minimax_model(), minimax_url(),
    );
    let msg = anthropic_rs::ChatMessage::user_with_blocks(vec![
        anthropic_rs::ContentBlock::Image {
            source: anthropic_rs::ImageSource {
                source_type: "url".into(),
                media_type: "image/png".into(),
                data: "https://upload.wikimedia.org/wikipedia/commons/thumb/8/80/Wikipedia-logo-v2.svg/200px-Wikipedia-logo-v2.svg.png".into(),
            },
        },
        anthropic_rs::ContentBlock::Text {
            text: "Describe this image in one short sentence.".into(),
            citations: None,
        },
    ]);
    let resp = client.messages_create(&[msg], None, None, 512).expect("real image failed");
    println!("[minimax image] '{}' (len={})", resp.text.trim(), resp.text.len());
    assert!(!resp.text.is_empty(), "should describe the image");
    assert!(resp.text.len() > 20, "description too short: {}", resp.text.len());
}

// ── Async test ─────────────────────────────────────────────────────────────
#[cfg(feature = "async")]
#[tokio::test]
async fn minimax_async_basic_message() {
    if !should_run("MINIMAX") { return; }
    let client = anthropic_rs::AnthropicAsyncClient::with_base_url(
        minimax_key(), minimax_model(), minimax_url(),
    );
    let resp = client.messages_create(
        &[anthropic_rs::ChatMessage::user("Reply with just the word 'OK'.")],
        None, None, 128,
    ).await.expect("async messages_create failed");
    println!("[minimax async] message: {}", resp.text.trim());
    assert!(!resp.text.is_empty());
}