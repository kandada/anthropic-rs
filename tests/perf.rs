// Copyright (c) 2025 xiefujin <490021684@qq.com>
// Licensed under Apache-2.0, see LICENSE file for full license terms.

//! Performance / scale tests for anthropic-rs.
//!
//! These guard against pathological (e.g. O(n²)) regressions in the hot
//! paths: SSE streaming, fragmented tool-call assembly, thinking block
//! reconstruction, message building, token counting and retry backoff.
//! Thresholds are deliberately loose (10–50x expected) so they never flake
//! on slow CI, yet catch quadratic blowups that grow with input size.

use std::io::Cursor;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

use anthropic_client_rs::*;

fn parse_stream(raw: &str) -> LlmResponse {
    let cancel = AtomicBool::new(false);
    parse_anthropic_stream(Cursor::new(raw.as_bytes().to_vec()), |_| {}, |_, _| {}, &cancel).unwrap()
}

fn big_text_stream(n: usize) -> String {
    let mut s = String::with_capacity(n * 90 + 16);
    for _ in 0..n {
        s.push_str(
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"x\"}}\n\n",
        );
    }
    s.push_str("data: [DONE]\n\n");
    s
}

fn big_thinking_stream(n_blocks: usize) -> String {
    let mut s = String::with_capacity(n_blocks * 300);
    for i in 0..n_blocks {
        s.push_str(&format!(
            "data: {{\"type\":\"content_block_start\",\"index\":{i},\"content_block\":{{\"type\":\"thinking\"}}}}\n\n"
        ));
        s.push_str(&format!(
            "data: {{\"type\":\"content_block_delta\",\"index\":{i},\"delta\":{{\"type\":\"thinking_delta\",\"thinking\":\"step {i} \"}}}}\n\n"
        ));
        s.push_str(&format!(
            "data: {{\"type\":\"content_block_delta\",\"index\":{i},\"delta\":{{\"type\":\"signature_delta\",\"signature\":\"sig{i}\"}}}}\n\n"
        ));
    }
    s.push_str("data: [DONE]\n\n");
    s
}

fn big_tool_stream(n_blocks: usize) -> String {
    let mut s = String::with_capacity(n_blocks * 400);
    for i in 0..n_blocks {
        s.push_str(&format!(
            "data: {{\"type\":\"content_block_start\",\"index\":{i},\"content_block\":{{\"type\":\"tool_use\",\"id\":\"t{i}\",\"name\":\"run\"}}}}\n\n"
        ));
        let arg = format!("{{\"cmd\":\"echo {i}\"}}");
        // split args into 4 fragments
        let mid1 = arg.len() / 4;
        let mid2 = arg.len() / 2;
        let mid3 = (3 * arg.len()) / 4;
        for (a, b) in [(0, mid1), (mid1, mid2), (mid2, mid3), (mid3, arg.len())] {
            s.push_str(&format!(
                "data: {{\"type\":\"content_block_delta\",\"index\":{i},\"delta\":{{\"type\":\"input_json_delta\",\"partial_json\":\"{}\"}}}}\n\n",
                escape_json(&arg[a..b])
            ));
        }
    }
    s.push_str("data: [DONE]\n\n");
    s
}

fn escape_json(s: &str) -> String {
    // Only quotes/backslashes need escaping for our synthetic args.
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

#[test]
fn streams_100k_text_deltas() {
    let raw = big_text_stream(100_000);
    let start = Instant::now();
    let resp = parse_stream(&raw);
    let elapsed = start.elapsed();
    assert_eq!(resp.text.len(), 100_000);
    assert!(elapsed < Duration::from_secs(5), "100k deltas took {elapsed:?}");
    println!("100k text deltas parsed in {elapsed:?}");
}

#[test]
fn assembles_many_thinking_blocks_with_signatures() {
    let raw = big_thinking_stream(2_000);
    let start = Instant::now();
    let resp = parse_stream(&raw);
    let elapsed = start.elapsed();
    let thinks = resp
        .content_blocks
        .iter()
        .filter(|b| matches!(b, ContentBlock::Thinking { .. }))
        .count();
    assert_eq!(thinks, 2_000);
    // Every signature must be preserved in order.
    for (i, cb) in resp
        .content_blocks
        .iter()
        .filter(|b| matches!(b, ContentBlock::Thinking { .. }))
        .enumerate()
    {
        match cb {
            ContentBlock::Thinking { signature, .. } => {
                assert_eq!(signature, &format!("sig{i}"));
            }
            _ => unreachable!(),
        }
    }
    assert!(elapsed < Duration::from_secs(5), "2k thinking blocks took {elapsed:?}");
    println!("2k thinking blocks (thinking+signature deltas) in {elapsed:?}");
}

#[test]
fn assembles_many_fragmented_tool_calls() {
    let raw = big_tool_stream(500);
    let start = Instant::now();
    let resp = parse_stream(&raw);
    let elapsed = start.elapsed();
    assert_eq!(resp.tool_calls.len(), 500);
    for (i, tc) in resp.tool_calls.iter().enumerate() {
        assert_eq!(tc.name, "run");
        assert_eq!(tc.parsed_args()["cmd"], format!("echo {i}"));
    }
    assert!(elapsed < Duration::from_secs(5), "500 tool calls took {elapsed:?}");
    println!("500 fragmented tool calls assembled in {elapsed:?}");
}

#[test]
fn builds_large_message_history() {
    let mut msgs = Vec::with_capacity(2_000);
    for i in 0..2_000 {
        match i % 3 {
            0 => msgs.push(ChatMessage::user(format!("user message {i} with content here"))),
            1 => msgs.push(ChatMessage::assistant(format!("assistant reply {i}"))),
            _ => msgs.push(ChatMessage::tool_result("tool_abc", "tool output")),
        }
    }
    let start = Instant::now();
    let out = AnthropicClient::build_anthropic_messages(&msgs);
    let elapsed = start.elapsed();
    assert!(!out.is_empty());
    assert!(elapsed < Duration::from_secs(5), "2k-message history took {elapsed:?}");
    println!("2k-message history serialized in {elapsed:?}");
}

#[test]
fn counts_tokens_on_large_text() {
    let big = "the quick brown fox jumps over the lazy dog ".repeat(20_000);
    let start = Instant::now();
    let n = anthropic_client_rs::tokens::count_system_tokens(&big);
    let elapsed = start.elapsed();
    assert!(n > 0);
    assert!(elapsed < Duration::from_secs(5), "1MB token count took {elapsed:?}");
    println!("1MB token count ({n} tokens) in {elapsed:?}");
}

#[test]
fn retry_delay_many_invocations() {
    use anthropic_client_rs::retry::RetryConfig;
    let cfg = RetryConfig::default();
    let start = Instant::now();
    let mut acc = 0u64;
    for attempt in 0..100_000u32 {
        acc = acc.wrapping_add(cfg.delay_ms(attempt % 32));
    }
    let elapsed = start.elapsed();
    assert!(acc > 0);
    assert!(elapsed < Duration::from_secs(2), "100k delay_ms took {elapsed:?}");
    println!("100k retry delay computations in {elapsed:?}");
}

#[test]
fn non_streaming_assemble_large_response() {
    let mut blocks = Vec::new();
    for i in 0..5_000 {
        blocks.push(serde_json::json!({"type": "text", "text": format!("block {i} ")}));
    }
    let raw = serde_json::json!({
        "id": "m", "type": "message", "role": "assistant",
        "content": blocks, "model": "claude",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 2}
    });
    let start = Instant::now();
    let resp = anthropic_client_rs::api_common::assemble_message_response(&raw);
    let elapsed = start.elapsed();
    assert_eq!(resp.content_blocks.len(), 5_000);
    assert!(elapsed < Duration::from_secs(5), "5k-block assembly took {elapsed:?}");
    println!("5k-block response assembled in {elapsed:?}");
}
