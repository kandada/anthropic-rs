// Copyright (c) 2025 xiefujin <490021684@qq.com>
// Licensed under Apache-2.0, see LICENSE file for full license terms.

//! Protocol-specific end-to-end test for anthropic-rs.
//!
//! Anthropic's distinctive semantics exercised here:
//!   - **Extended thinking**: `thinking` blocks close with a
//!     `signature_delta`; the signature is required to pass the block back
//!     in the next turn.
//!   - **Multi-turn rule**: when thinking + tool use, the assistant turn
//!     must include the thinking blocks (with signatures) — dropping them
//!     is rejected by real Anthropic (some gateways are lenient).
//!   - **Tool use**: `tool_use` content blocks carry a JSON-object `input`;
//!     tool results are `tool_result` blocks coalesced into a `user` turn.
//!   - **Streaming**: `thinking_delta` / `signature_delta` /
//!     `input_json_delta` arrive as content_block_delta events.
//!
//! Credentials come from the environment — never hard-coded:
//!   LLM_API_KEY / LLM_API_URL / LLM_MODEL_NAME / [LLM_THINKING]
//!
//! Run:
//!   export LLM_API_KEY=... LLM_API_URL=... LLM_MODEL_NAME=...
//!   cargo run --example e2e

use std::env;
use std::process::exit;

use anthropic_client_rs::*;
use serde_json::json;

fn env(name: &str) -> String {
    env::var(name).unwrap_or_else(|_| {
        eprintln!("missing required env var: {name}");
        exit(2);
    })
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        format!("{}...", s.chars().take(n).collect::<String>())
    }
}

/// Tiny assertion harness: real failures exit non-zero; provider quirks are
/// reported as notes so a lenient/chatty model doesn't fail the suite.
struct T {
    passed: u32,
    failed: Vec<String>,
}

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

fn thinking_blocks(resp: &LlmResponse) -> Vec<&ContentBlock> {
    resp.content_blocks
        .iter()
        .filter(|b| matches!(b, ContentBlock::Thinking { .. }))
        .collect()
}

fn describe_blocks(resp: &LlmResponse) {
    for cb in &resp.content_blocks {
        match cb {
            ContentBlock::Thinking { thinking, signature } => {
                println!(
                    "  [thinking] {} chars, signature={} chars",
                    thinking.chars().count(),
                    signature.chars().count()
                );
            }
            ContentBlock::RedactedThinking { data } => {
                println!("  [redacted_thinking] {} chars", data.chars().count());
            }
            ContentBlock::ToolUse { id, name, input } => {
                println!(
                    "  [tool_use] id={id} name={name} input={}",
                    serde_json::to_string(input).unwrap_or_default()
                );
            }
            ContentBlock::Text { text, .. } => println!("  [text] {}", truncate(text, 80)),
            other => println!("  [{other:?}]"),
        }
    }
}

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

fn main() {
    let key = env("LLM_API_KEY");
    let base = env("LLM_API_URL");
    let model = env("LLM_MODEL_NAME");
    let thinking_mode = env::var("LLM_THINKING").ok().filter(|s| !s.is_empty());
    let thinking: Option<ThinkingConfig> = match thinking_mode.as_deref() {
        Some("adaptive") => Some(ThinkingConfig::adaptive()),
        Some("enabled") => Some(ThinkingConfig::enabled(4096)),
        _ => None,
    };

    println!("== anthropic-rs e2e (protocol-specific) ==");
    println!("  endpoint: {base}");
    println!("  model:    {model}");
    println!("  thinking: {}", thinking_mode.as_deref().unwrap_or("none"));

    let client = AnthropicClient::with_base_url(&key, &model, &base).with_total_timeout(120);
    let tools = make_tools();
    let system = "You are a helpful assistant. When asked about the weather, call get_weather for each requested city and wait for the result.";

    let mut t = T { passed: 0, failed: Vec::new() };

    // ── Test 1: thinking capture (non-streaming) ───────────────────────────
    println!("── Test 1: thinking capture ──");
    let first = client
        .messages_create(
            &[ChatMessage::user("What is the weather in San Francisco? Use the tool.")],
            Some(system),
            Some(&tools),
            2048,
            thinking.as_ref(),
        )
        .unwrap_or_else(|e| {
            eprintln!("❌ test 1 request failed: {e}");
            exit(1);
        });
    describe_blocks(&first);
    let thinks = thinking_blocks(&first);
    if thinking.is_some() {
        t.check(!thinks.is_empty(), "thinking block(s) captured when thinking enabled");
        if let Some(ContentBlock::Thinking { signature, .. }) = thinks.first() {
            t.check(!signature.is_empty(), "thinking block carries a non-empty signature");
        }
        t.check(
            first.reasoning_content.is_some(),
            "reasoning_content populated from thinking deltas",
        );
    } else {
        t.note("thinking disabled — skipping thinking assertions");
    }
    t.check(!first.tool_calls.is_empty(), "model produced a tool_use");
    for tc in &first.tool_calls {
        t.check(!tc.id.is_empty(), &format!("tool_use has id ({})", tc.id));
        t.check(!tc.name.is_empty(), "tool_use has name");
        t.check(tc.parsed_args().is_object(), "tool_use input is a JSON object");
    }

    // ── Test 2: multi-turn tool loop WITH thinking roundtrip ───────────────
    // Pass the assistant turn back including the thinking blocks (signature),
    // then the tool_result blocks — exactly what real Anthropic requires.
    println!("── Test 2: multi-turn with thinking roundtrip ──");
    let mut history: Vec<ChatMessage> = vec![ChatMessage::user(
        "What is the weather in San Francisco and in Paris? Use the tool for each.",
    )];
    let mut rounds_ok = 0;
    for round in 0..3 {
        let resp = client
            .messages_create(&history, Some(system), Some(&tools), 2048, thinking.as_ref())
            .unwrap_or_else(|e| {
                eprintln!("❌ multi-turn round {round} failed: {e}");
                exit(1);
            });
        if resp.tool_calls.is_empty() {
            println!("  round {round}: finished with text: {}", truncate(&resp.text, 120));
            rounds_ok += 1;
            break;
        }
        // Roundtrip the full block list (thinking + tool_use + text).
        let mut asst = ChatMessage::assistant(resp.text.clone());
        asst.content_blocks = Some(resp.content_blocks.clone());
        history.push(asst);
        for tc in &resp.tool_calls {
            let city = tc.parsed_args().get("city").cloned().unwrap_or_else(|| json!("?"));
            history.push(ChatMessage::tool_result(&tc.id, format!("Weather in {city}: sunny, 22C")));
        }
        rounds_ok += 1;
    }
    t.check(rounds_ok > 0, "multi-turn loop completed without 400 (thinking roundtripped)");

    // ── Test 3: prove the signature requirement ────────────────────────────
    // Repeat the SAME conversation but strip the thinking blocks from the
    // assistant turn. Real Anthropic rejects this; some gateways are lenient.
    println!("── Test 3: dropping signature (expect rejection on strict gateways) ──");
    if let Some(first_tcs) = first.tool_calls.first() {
        let full = ToolCall {
            id: first_tcs.id.clone(),
            call_type: "tool_use".into(),
            name: first_tcs.name.clone(),
            input: first_tcs.parsed_args(),
        };
        let stripped: Vec<ChatMessage> = vec![
            ChatMessage::user("What is the weather in San Francisco? Use the tool."),
            ChatMessage::assistant_with_tools("", vec![full]),
            ChatMessage::tool_result(&first_tcs.id, "Weather: sunny, 22C"),
        ];
        match client.messages_create(&stripped, Some(system), Some(&tools), 2048, thinking.as_ref()) {
            Ok(_) => t.note("provider ACCEPTED assistant turn without thinking/signature (lenient gateway)"),
            Err(e) => t.note(&format!("provider REJECTED missing thinking/signature as expected: {e}")),
        }
    } else {
        t.note("no tool call to strip — skipping signature-requirement probe");
    }

    // ── Test 4: streaming thinking + tool use ──────────────────────────────
    println!("── Test 4: streaming (thinking_delta + input_json_delta) ──");
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
        .unwrap_or_else(|e| {
            eprintln!("❌ streaming failed: {e}");
            exit(1);
        });
    if thinking.is_some() {
        if let Some(rc) = &resp.reasoning_content {
            streamed_reasoning.push_str(rc);
        }
        t.check(!streamed_reasoning.is_empty(), "streamed thinking_delta captured as reasoning");
    } else {
        t.note("thinking disabled — skipping streamed thinking assertion");
    }
    t.note(&format!("streamed text length: {}", streamed_text.chars().count()));
    t.note(&format!("streamed tool calls: {streamed_tool_calls}"));
    if let Some(u) = &resp.usage {
        println!(
            "  usage: in={} out={} cache_read={:?} cache_creation={:?}",
            u.input_tokens, u.output_tokens, u.cache_read_input_tokens, u.cache_creation_input_tokens
        );
        t.check(u.output_tokens > 0, "streaming usage captured from message_delta");
    }

    // ── Test 5: MessageRequest full path (adaptive thinking) ───────────────
    println!("── Test 5: MessageRequest builder (thinking=adaptive) ──");
    let req = MessageRequest::new(&model, vec![ChatMessage::user("Say hello.")], 1024)
        .system(system)
        .tools(tools.clone())
        .thinking(ThinkingConfig::adaptive());
    match client.messages_send(&req) {
        Ok(resp) => t.check(!resp.text.is_empty(), "messages_send returns text"),
        Err(e) => t.check(false, &format!("messages_send failed: {e}")),
    }

    // ── Test 6: typed stream events + count_tokens endpoint ────────────────
    println!("── Test 6: typed event stream + count_tokens ──");
    match client.count_tokens(&[ChatMessage::user("hello world")], Some(system), Some(&tools)) {
        Ok(ct) => {
            println!("  count_tokens input_tokens={}", ct.input_tokens);
            t.check(ct.input_tokens > 0, "count_tokens endpoint returns a token count");
        }
        Err(e) => t.check(false, &format!("count_tokens failed: {e}")),
    }
    match client.messages_stream_events(&[ChatMessage::user("Say hi.")], Some(system), None, 1024, None) {
        Ok(mut events) => {
            let mut kinds = Vec::new();
            let mut last = events.next();
            while let Some(ev) = last {
                match ev {
                    Ok(e) => kinds.push(e.event_type),
                    Err(e) => {
                        t.check(false, &format!("typed stream error: {e}"));
                        break;
                    }
                }
                last = events.next();
            }
            t.check(!kinds.is_empty(), &format!("typed event stream yielded {:?}", kinds));
            t.check(kinds.contains(&"message_stop".to_string()), "typed stream ended with message_stop");
        }
        Err(e) => t.check(false, &format!("messages_stream_events failed: {e}")),
    }

    t.finish();
    println!("✅ anthropic-rs e2e PASSED");
}
