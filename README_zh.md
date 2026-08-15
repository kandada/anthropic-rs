<!-- Copyright (c) 2025 xiefujin <490021684@qq.com> -->
<!-- Licensed under Apache-2.0, see LICENSE file for full license terms. -->


# anthropic-rs

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

全功能 Rust Anthropic API 客户端。兼容 Claude、MiniMax、DeepSeek-anthropic 等**任何 Anthropic 兼容的 API 提供商**。

[English](README.md)

## 特性

- **同步 + 异步** — 轻量 `ureq`（默认）或 `reqwest` + `tokio`（feature `async`）
- **Messages API** — 流式 SSE，content block delta 回调
- **工具调用** — 并行 tool calling，分段 JSON 累积
- **扩展推理** — thinking 内容块，支持推理链
- **提示缓存** — cache_control 断点，节省成本
- **自动重试** — 指数退避 + 抖动（同步 + 异步）
- **Token 计数** — 近似消息 token 估算
- **视觉识别** — 图片输入（base64 / URL）
- **系统提示** — 顶层 `system` 参数
- **多提供商** — 根据 API key 自动识别，自定义 base URL

## 安装

```toml
[dependencies]
anthropic-rs = "0.1"

# 异步支持
# anthropic-rs = { version = "0.1", features = ["async"] }
```

## 快速开始

### 同步

```rust
use anthropic_rs::{AnthropicClient, ChatMessage};

let client = AnthropicClient::new("sk-ant-xxx", "claude-sonnet-4-20250514");
let resp = client.messages_create(
    &[ChatMessage::user("Rust 的特点？")],
    Some("你是帮助性助手。"), None, 1024,
).unwrap();
println!("{}", resp.text);
```

### 异步

```rust
use anthropic_rs::{AnthropicAsyncClient, ChatMessage};

let client = AnthropicAsyncClient::new("sk-ant-xxx", "claude-sonnet-4-20250514");
let resp = client.messages_create(&[ChatMessage::user("你好")], Some("简洁回答"), None, 1024).await.unwrap();
```

## 流式输出

```rust
client.messages_stream(
    &[ChatMessage::user("讲个故事")],
    Some("你有创造力。"), None, 1024,
    |delta| { print!("{delta}"); },
    |tool_name, tool_args| { println!("工具: {tool_name}"); },
).unwrap();
```

## 工具调用

```rust
use anthropic_rs::Tool;
use serde_json::json;

let tools = &[Tool::new("get_weather", "查询天气",
    json!({"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}),
)];

let resp = client.messages_create(
    &[ChatMessage::user("北京天气？")],
    None, Some(tools), 1024,
).unwrap();

for tc in &resp.tool_calls {
    println!("{} -> {}", tc.name, tc.parsed_args()["location"]);
}
```

## 请求构建器

```rust
use anthropic_rs::{MessageRequest, ThinkingConfig, Metadata};

let req = MessageRequest::new("claude-sonnet-4-20250514", messages, 2048)
    .system("你是帮助性助手。")
    .temperature(0.7).top_p(0.9).top_k(40)
    .stop_sequences(vec!["END".into()])
    .tools(tools)
    .thinking(ThinkingConfig::enabled(4096))
    .metadata(Metadata { user_id: Some("user-1".into()) });

let body = req.build_body();
let resp = client.post_json(body).unwrap();
```

## 提示缓存

```rust
use anthropic_rs::CacheConfig;

let config = CacheConfig::standard();  // 缓存 system + 最后工具 + 最后 2 条消息
let mut body = req.build_body();
config.apply(&mut body);
// 后续相同前缀的请求会复用缓存，节省 90% 输入成本
```

## 自动重试

```rust
use anthropic_rs::{RetryConfig, retry};
let config = RetryConfig { max_retries: 3, ..Default::default() };
let result = retry_sync(|| client.messages_create(&[...], None, None, 1024), &config, |e| e.is_retryable());
```

## Token 计数

```rust
use anthropic_rs::tokens;
let n = tokens::count_message_tokens(&msg);
let total = tokens::count_messages_tokens(&messages);
```

## 视觉识别

```rust
use anthropic_rs::{ChatMessage, ContentBlock, ImageSource};

let msg = ChatMessage::user_with_blocks(vec![
    ContentBlock::Image {
        source: ImageSource { source_type: "base64".into(), media_type: "image/jpeg".into(), data: "/9j/...".into() },
    },
    ContentBlock::Text { text: "描述这张图。".into(), citations: None },
]);
let resp = client.messages_create(&[msg], None, None, 1024).unwrap();
```

## 多提供商

```rust
let client = AnthropicClient::new("sk-ant-xxx", "claude-sonnet-4-20250514");
let client = AnthropicClient::with_base_url("sk-xxx", "MiniMax-M2.7", "https://api.minimax.chat/anthropic");
let client = AnthropicClient::with_base_url("sk-xxx", "deepseek-chat", "https://api.deepseek.com/anthropic");
```

## Anthropic 特有功能

- **顶层系统提示** — `system` 独立于消息数组
- **内容块结构** — `text`、`image`、`tool_use`、`tool_result`、`thinking`、`document`
- **并行工具调用** — 一条 assistant 消息可包含多个 `tool_use` 块
- **停止原因** — `end_turn`、`max_tokens`、`tool_use`、`stop_sequence`
- **扩展推理** — `thinking` 块带 `signature` 签名

## 错误处理

```rust
use anthropic_rs::AnthropicError;

match client.messages_create(&[...], None, None, 1024) {
    Ok(resp) => println!("{}", resp.text),
    Err(AnthropicError::Api(msg)) => eprintln!("API: {msg}"),
    Err(AnthropicError::Network(msg)) => eprintln!("网络: {msg}"),
    Err(e) => eprintln!("错误: {e}"),
}

if err.is_retryable() { /* 重试 */ }
```

## 许可证

Apache-2.0