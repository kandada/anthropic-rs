// Copyright (c) 2025 xiefujin <490021684@qq.com>
// Licensed under Apache-2.0, see LICENSE file for full license terms.

//! Integration tests for new anthropic-rs modules: retry, tokens, cache.

use anthropic_client_rs::*;

// ── Retry ──────────────────────────────────────────────────────────────────

#[test]
fn test_retry_succeeds_on_eventual_success() {
    use anthropic_client_rs::retry::*;
    let config = RetryConfig { max_retries: 5, base_delay_ms: 1, max_delay_ms: 10 };
    let mut calls = 0;
    let result: std::result::Result<i32, &str> = retry_sync(
        || { calls += 1; if calls < 4 { Err("transient") } else { Ok(42) } },
        &config, |_| true,
    );
    assert_eq!(result.unwrap(), 42);
    assert_eq!(calls, 4);
}

#[test]
fn test_retry_stops_at_max() {
    use anthropic_client_rs::retry::*;
    let config = RetryConfig { max_retries: 2, base_delay_ms: 1, max_delay_ms: 10 };
    let mut calls = 0;
    let result: std::result::Result<i32, &str> = retry_sync(
        || { calls += 1; Err("always fail") },
        &config, |_| true,
    );
    assert!(result.is_err());
    assert_eq!(calls, 3);
}

#[test]
fn test_retry_non_retryable_immediate() {
    use anthropic_client_rs::retry::*;
    let config = RetryConfig::default();
    let mut calls = 0;
    let result: std::result::Result<i32, &str> = retry_sync(
        || { calls += 1; Err("fatal") },
        &config, |_| false,
    );
    assert!(result.is_err());
    assert_eq!(calls, 1);
}

#[test]
fn test_retry_with_anthropic_error() {
    use anthropic_client_rs::retry::*;
    let config = RetryConfig { max_retries: 2, base_delay_ms: 1, max_delay_ms: 10 };
    let mut calls = 0;
    let result: std::result::Result<i32, AnthropicError> = retry_sync(
        || {
            calls += 1;
            if calls < 2 { Err(AnthropicError::Network("timeout".into())) }
            else { Ok(42) }
        },
        &config, |e| e.is_retryable(),
    );
    assert_eq!(result.unwrap(), 42);
}

// ── Token Counting ─────────────────────────────────────────────────────────

#[test]
fn test_count_tokens_english() {
    let n = anthropic_client_rs::tokens::count_message_tokens(&ChatMessage::user("Hello world!"));
    assert!(n >= 5, "got {n}");
}

#[test]
fn test_count_tokens_system() {
    let n = anthropic_client_rs::tokens::count_system_tokens("You are a helpful assistant.");
    assert!(n >= 8, "got {n}");
}

#[test]
fn test_count_tokens_empty_system() {
    assert_eq!(anthropic_client_rs::tokens::count_system_tokens(""), 0);
}

#[test]
fn test_count_messages_tokens() {
    let msgs = vec![
        ChatMessage::user("Hello"),
        ChatMessage::assistant("Hi there!"),
    ];
    let n = anthropic_client_rs::tokens::count_messages_tokens(&msgs);
    assert!(n >= 10, "got {n}");
}

#[test]
fn test_count_with_image_block() {
    let msg = ChatMessage::user_with_blocks(vec![
        ContentBlock::Text { text: "hi".into(), citations: None },
        ContentBlock::Image { source: ImageSource {
            source_type: "base64".into(), media_type: "image/png".into(), data: "xxx".into(),
        }},
    ]);
    let n = anthropic_client_rs::tokens::count_message_tokens(&msg);
    assert!(n > 100, "image should cost tokens, got {n}");
}

// ── Cache ──────────────────────────────────────────────────────────────────

#[test]
fn test_cache_system_prompt() {
    let mut body = serde_json::json!({"system": "You are helpful."});
    anthropic_client_rs::cache::cache_system(&mut body);
    let sys = &body["system"];
    assert!(sys.is_array());
    assert_eq!(sys[0]["type"], "text");
    assert_eq!(sys[0]["cache_control"]["type"], "ephemeral");
}

#[test]
fn test_cache_last_tool() {
    let mut body = serde_json::json!({
        "tools": [
            {"name": "a", "input_schema": {"type": "object"}},
            {"name": "b", "input_schema": {"type": "object"}}
        ]
    });
    anthropic_client_rs::cache::cache_last_tool(&mut body);
    assert_eq!(body["tools"][1]["cache_control"]["type"], "ephemeral");
    // First tool should NOT have cache_control
    assert!(body["tools"][0].get("cache_control").is_none());
}

#[test]
fn test_cache_last_message_block() {
    let mut msg = serde_json::json!({
        "role": "user",
        "content": [
            {"type": "text", "text": "first"},
            {"type": "text", "text": "last"}
        ]
    });
    anthropic_client_rs::cache::cache_last_message_block(&mut msg);
    let content = msg["content"].as_array().unwrap();
    assert!(content[0].get("cache_control").is_none());
    assert_eq!(content[1]["cache_control"]["type"], "ephemeral");
}

#[test]
fn test_cache_config_standard() {
    let config = CacheConfig::standard();
    assert!(config.cache_system);
    assert!(config.cache_tools);
    assert_eq!(config.cache_last_messages, 2);
}

#[test]
fn test_cache_config_apply() {
    let mut body = serde_json::json!({
        "system": "You are helpful.",
        "tools": [{"name": "f1", "input_schema": {}}, {"name": "f2", "input_schema": {}}],
        "messages": [
            {"role": "user", "content": "msg1"},
            {"role": "user", "content": "msg2"},
            {"role": "user", "content": "msg3"},
            {"role": "user", "content": [{"type": "text", "text": "msg4"}]}
        ]
    });
    CacheConfig::standard().apply(&mut body);
    // System cached
    assert!(body["system"][0].get("cache_control").is_some());
    // Last tool cached
    assert!(body["tools"][1].get("cache_control").is_some());
    // Last 2 messages cached (msg3, msg4)
    assert!(body["messages"][2]["content"].get("cache_control").is_none()); // msg3 is string, not array
    let msg4 = &body["messages"][3];
    assert_eq!(msg4["content"].as_array().unwrap().last().unwrap()["cache_control"]["type"], "ephemeral");
}