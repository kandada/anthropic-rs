// Copyright (c) 2025 xiefujin <490021684@qq.com>
// Licensed under Apache-2.0, see LICENSE file for full license terms.

//! Async end-to-end test for anthropic-rs (`AnthropicAsyncClient`).
//!
//! Same protocol-specific coverage as `e2e.rs` but through the async
//! `reqwest` client: thinking capture, multi-turn tool loop with thinking
//! roundtrip, and streaming (thinking_delta + input_json_delta + usage).
//!
//! Credentials come from the environment — never hard-coded:
//!   LLM_API_KEY / LLM_API_URL / LLM_MODEL_NAME / [LLM_THINKING]
//!
//! Run:
//!   export LLM_API_KEY=... LLM_API_URL=... LLM_MODEL_NAME=... [LLM_THINKING=adaptive]
//!   cargo run --example e2e_async --features async

#[cfg(feature = "async")]
use std::env;
use std::process::exit;
#[cfg(feature = "async")]
use anthropic_client_rs::*;
#[cfg(feature = "async")]
use serde_json::json;

#[cfg(not(feature = "async"))]
fn main() {
    eprintln!("this example requires the `async` feature: cargo run --example e2e_async --features async");
    exit(2);
}

#[cfg(feature = "async")]
fn env(name: &str) -> String {
    env::var(name).unwrap_or_else(|_| {
        eprintln!("missing required env var: {name}");
        exit(2);
    })
}

#[cfg(feature = "async")]
fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        format!("{}...", s.chars().take(n).collect::<String>())
    }
}

#[cfg(feature = "async")]
struct T {
    passed: u32,
    failed: Vec<String>,
}

#[cfg(feature = "async")]
impl T {
    fn check(&mut self, ok: bool, label: &str) {
        if ok {
            self.passed += 1;
            println!("  ✅ {label}");
        } else {
            self.failed.push(label.to_string());
            println!("  ❌ {label}");
        }
    }
    fn note(&self, label: &str) {
        println!("  ⚠️  {label}");
    }
    fn finish(self) {
        println!("\n  passed={} failed={}", self.passed, self.failed.len());
        if !self.failed.is_empty() {
            for f in &self.failed {
                eprintln!("  FAILED: {f}");
            }
            exit(1);
        }
    }
}

#[cfg(feature = "async")]
fn make_tools() -> Vec<Tool> {
    vec![Tool::new(
        "get_weather",
        "Get the current weather for a city.",
        json!({
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
        }),
    )]
}

#[cfg(feature = "async")]
#[tokio::main]
async fn main() {
    let key = env("LLM_API_KEY");
    let base = env("LLM_API_URL");
    let model = env("LLM_MODEL_NAME");
    let thinking_mode = env::var("LLM_THINKING").ok().filter(|s| !s.is_empty());
    let thinking: Option<ThinkingConfig> = match thinking_mode.as_deref() {
        Some("adaptive") => Some(ThinkingConfig::adaptive()),
        Some("enabled") => Some(ThinkingConfig::enabled(4096)),
        _ => None,
    };

    println!("== anthropic-rs e2e (ASYNC) ==");
    println!("  endpoint: {base}");
    println!("  model:    {model}");
    println!("  thinking: {}", thinking_mode.as_deref().unwrap_or("none"));

    let client = AnthropicAsyncClient::with_base_url(&key, &model, &base);
    let tools = make_tools();
    let system = "You are a helpful assistant. When asked about the weather, call get_weather for each requested city and wait for the result.";
    let mut t = T { passed: 0, failed: Vec::new() };

    // ── Test 1: async thinking capture ─────────────────────────────────────
    println!("── Test 1: async thinking capture ──");
    let first = client
        .messages_create(
            &[ChatMessage::user("What is the weather in San Francisco? Use the tool.")],
            Some(system),
            Some(&tools),
            2048,
            thinking.as_ref(),
        )
        .await
        .unwrap_or_else(|e| {
            eprintln!("❌ test 1 failed: {e}");
            exit(1);
        });
    for cb in &first.content_blocks {
        match cb {
            ContentBlock::Thinking { thinking, signature } => println!(
                "  [thinking] {} chars, signature={} chars",
                thinking.chars().count(),
                signature.chars().count()
            ),
            ContentBlock::ToolUse { id, name, input } => println!(
                "  [tool_use] id={id} name={name} input={}",
                serde_json::to_string(input).unwrap_or_default()
            ),
            _ => {}
        }
    }
    if thinking.is_some() {
        let has_thinking = first
            .content_blocks
            .iter()
            .any(|b| matches!(b, ContentBlock::Thinking { signature, .. } if !signature.is_empty()));
        t.check(has_thinking, "async thinking block captured with signature");
    } else {
        t.note("thinking disabled — skipping");
    }
    t.check(!first.tool_calls.is_empty(), "async model produced a tool_use");

    // ── Test 2: async multi-turn tool loop with thinking roundtrip ─────────
    println!("── Test 2: async multi-turn with thinking roundtrip ──");
    let mut history: Vec<ChatMessage> = vec![ChatMessage::user(
        "What is the weather in San Francisco and in Paris? Use the tool for each.",
    )];
    let mut rounds_ok = 0;
    for round in 0..3 {
        let resp = client
            .messages_create(&history, Some(system), Some(&tools), 2048, thinking.as_ref())
            .await
            .unwrap_or_else(|e| {
                eprintln!("❌ async round {round} failed: {e}");
                exit(1);
            });
        if resp.tool_calls.is_empty() {
            println!("  round {round}: finished: {}", truncate(&resp.text, 120));
            rounds_ok += 1;
            break;
        }
        let mut asst = ChatMessage::assistant(resp.text.clone());
        asst.content_blocks = Some(resp.content_blocks.clone());
        history.push(asst);
        for tc in &resp.tool_calls {
            let city = tc.parsed_args().get("city").cloned().unwrap_or_else(|| json!("?"));
            history.push(ChatMessage::tool_result(&tc.id, format!("Weather in {city}: sunny, 22C")));
        }
        rounds_ok += 1;
    }
    t.check(rounds_ok > 0, "async multi-turn loop completed without 400");

    // ── Test 3: async streaming (thinking + tools + usage) ─────────────────
    println!("── Test 3: async streaming ──");
    let mut streamed_text = String::new();
    let mut streamed_reasoning = String::new();
    let mut streamed_tool_calls = 0usize;
    let resp = client
        .messages_stream(
            &[ChatMessage::user("Call get_weather for Tokyo immediately, then stop.")],
            Some(system),
            Some(&tools),
            2048,
            thinking.as_ref(),
            |d| streamed_text.push_str(d),
            |name, args| {
                streamed_tool_calls += 1;
                println!("  ↳ [stream] tool {name}: {args}");
            },
        )
        .await
        .unwrap_or_else(|e| {
            eprintln!("❌ async streaming failed: {e}");
            exit(1);
        });
    if let Some(rc) = &resp.reasoning_content {
        streamed_reasoning.push_str(rc);
    }
    if thinking.is_some() {
        t.check(!streamed_reasoning.is_empty(), "async streamed thinking captured");
    }
    println!("  streamed text: {}", truncate(&streamed_text, 120));
    println!("  streamed tool calls: {streamed_tool_calls}");
    if let Some(u) = &resp.usage {
        println!(
            "  usage: in={} out={} cache_read={:?} cache_creation={:?}",
            u.input_tokens, u.output_tokens, u.cache_read_input_tokens, u.cache_creation_input_tokens
        );
        t.check(u.output_tokens > 0, "async streaming usage captured");
    }

    // ── Test 4: async MessageRequest path ──────────────────────────────────
    println!("── Test 4: async messages_send ──");
    let req = MessageRequest::new(&model, vec![ChatMessage::user("Say hello.")], 1024)
        .system(system)
        .tools(tools)
        .thinking(ThinkingConfig::adaptive());
    match client.messages_send(&req).await {
        Ok(resp) => t.check(!resp.text.is_empty(), "async messages_send returns text"),
        Err(e) => t.check(false, &format!("async messages_send failed: {e}")),
    }

    t.finish();
    println!("✅ anthropic-rs async e2e PASSED");
}
