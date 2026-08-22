<!-- Copyright (c) 2025 xiefujin <490021684@qq.com> -->
<!-- Licensed under Apache-2.0, see LICENSE file for full license terms. -->


# anthropic-client-rs

[![Crates.io](https://img.shields.io/crates/v/anthropic-client-rs.svg)](https://crates.io/crates/anthropic-client-rs)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

A full-featured Rust client for the Anthropic Messages API. Compatible with Claude, MiniMax, DeepSeek-anthropic, and **any Anthropic-compatible provider**.

[中文文档](README_zh.md)

## Features

- **Sync + Async** — lightweight `ureq` (default) or `reqwest` + `tokio` (feature `async`)
- **Messages API** — streaming SSE with content block deltas
- **Tool Use** — parallel tool calling with partial JSON accumulation
- **Extended Thinking** — thinking blocks with reasoning support
- **Prompt Caching** — cache_control breakpoints for cost savings
- **Auto-Retry** — exponential backoff with jitter (sync + async)
- **Token Counting** — approximate message token estimation
- **Vision** — image inputs (base64, URL)
- **System Prompts** — top-level `system` parameter
- **Multi-Provider** — auto-detects provider by API key, custom base URL

## Quick Start

### Sync

```rust
use anthropic_client_rs::{AnthropicClient, ChatMessage};

let client = AnthropicClient::new("sk-ant-xxx", "claude-sonnet-4-20250514");
let resp = client.messages_create(
    &[ChatMessage::user("What is Rust?")],
    Some("You are helpful."), None, 1024,
).unwrap();
println!("{}", resp.text);
```

### Async

```rust
use anthropic_client_rs::{AnthropicAsyncClient, ChatMessage};

let client = AnthropicAsyncClient::new("sk-ant-xxx", "claude-sonnet-4-20250514");
let resp = client.messages_create(&[ChatMessage::user("Hi")], Some("Be brief"), None, 1024).await.unwrap();
```

## Streaming

```rust
client.messages_stream(
    &[ChatMessage::user("Tell a story")],
    Some("You are creative."), None, 1024,
    |delta| { print!("{delta}"); },           // text/thinking tokens
    |tool_name, tool_args| {                  // completed tool calls
        println!("Tool: {tool_name}");
    },
).unwrap();
```

## Tool Use

```rust
use anthropic_client_rs::{Tool};
use serde_json::json;

let tools = &[Tool::new("get_weather", "Get weather",
    json!({"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}),
)];

let resp = client.messages_create(
    &[ChatMessage::user("Weather in Paris?")],
    None, Some(tools), 1024,
).unwrap();

for tc in &resp.tool_calls {
    println!("{} -> {}", tc.name, tc.parsed_args()["location"]);
}
```

## Request Builder (Full API)

```rust
use anthropic_client_rs::{MessageRequest, ThinkingConfig, Metadata};

let req = MessageRequest::new("claude-sonnet-4-20250514", messages, 2048)
    .system("You are helpful.")
    .temperature(0.7)
    .top_p(0.9)
    .top_k(40)
    .stop_sequences(vec!["END".into()])
    .tools(tools)
    .thinking(ThinkingConfig::enabled(4096))
    .metadata(Metadata { user_id: Some("user-1".into()) });

let body = req.build_body();
let resp = client.post_json(body).unwrap();
```

## Prompt Caching

```rust
use anthropic_client_rs::CacheConfig;

let config = CacheConfig::standard();  // cache system + last tool + last 2 messages
let mut body = req.build_body();
config.apply(&mut body);
// Sends with cache_control breakpoints — subsequent requests reuse cached tokens
```

## Auto-Retry

```rust
use anthropic_client_rs::{RetryConfig, retry};

let config = RetryConfig { max_retries: 3, ..Default::default() };
let result = retry_sync(
    || client.messages_create(&[...], None, None, 1024),
    &config,
    |e| e.is_retryable(),
);
```

## Token Counting

```rust
use anthropic_client_rs::tokens;
let n = tokens::count_message_tokens(&msg);
let total = tokens::count_messages_tokens(&messages);
let sys = tokens::count_system_tokens("You are helpful.");
```

## Vision

```rust
use anthropic_client_rs::{ChatMessage, ContentBlock, ImageSource};

let msg = ChatMessage::user_with_blocks(vec![
    ContentBlock::Image {
        source: ImageSource {
            source_type: "base64".into(),
            media_type: "image/jpeg".into(),
            data: "/9j/4AAQ...".into(),
        },
    },
    ContentBlock::Text { text: "Describe this.".into(), citations: None },
]);

let resp = client.messages_create(&[msg], None, None, 1024).unwrap();
```

## Multi-Turn with Tool Results

```rust
// First turn — model calls a tool
let resp = client.messages_create(
    &[ChatMessage::user("Weather in Tokyo?")],
    None, Some(tools), 1024,
).unwrap();

// Return tool result
let messages = &[
    ChatMessage::user("Weather in Tokyo?"),
    ChatMessage::assistant_with_tools("", resp.tool_calls_to_typed()),
    ChatMessage::tool_result("toolu_xxx", "Sunny, 25C"),
];
let final_resp = client.messages_create(messages, None, None, 1024).unwrap();
```

## Multi-Provider

```rust
let client = AnthropicClient::new("sk-ant-xxx", "claude-sonnet-4-20250514");
let client = AnthropicClient::with_base_url("sk-xxx", "MiniMax-M2.7", "https://api.minimax.chat/anthropic");
let client = AnthropicClient::with_base_url("sk-xxx", "deepseek-chat", "https://api.deepseek.com/anthropic");
```

## Anthropic-Specific Features

- **Top-level system prompt** — separate from message array
- **Content blocks** — `text`, `image`, `tool_use`, `tool_result`, `thinking`, `document`
- **Parallel tool calls** — multiple `tool_use` blocks in one assistant message
- **Stop reasons** — `end_turn`, `max_tokens`, `tool_use`, `stop_sequence`
- **Extended thinking** — `thinking` block with `signature`

## Error Handling

```rust
use anthropic_client_rs::AnthropicError;

match client.messages_create(&[...], None, None, 1024) {
    Ok(resp) => println!("{}", resp.text),
    Err(AnthropicError::Api(msg)) => eprintln!("API: {msg}"),
    Err(AnthropicError::Network(msg)) => eprintln!("Network: {msg}"),
    Err(e) => eprintln!("Error: {e}"),
}

if err.is_retryable() { /* retry */ }
```

## License

Apache-2.0