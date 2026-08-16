// Copyright (c) 2025 xiefujin <490021684@qq.com>
// Licensed under Apache-2.0, see LICENSE file for full license terms.

//! # anthropic-client-rs
//!
//! A Rust client for the Anthropic Messages API — streaming SSE, tool use,
//! extended thinking, and prompt caching. Compatible with Claude, MiniMax,
//! DeepSeek-anthropic, and any Anthropic-compatible provider.
//!
//! ## Features
//!
//! - **Sync** (default) — lightweight `ureq`
//! - **Async** — optional `reqwest` + `tokio` (`features = ["async"]`)
//! - **Messages API** — streaming SSE with content block deltas
//! - **Tool use** — parallel tool calling with partial JSON accumulation
//! - **Extended thinking** — thinking blocks with reasoning support
//! - **Vision** — image inputs (base64, URL)
//! - **System prompts** — top-level `system` parameter
//! - **Prompt caching** — cache_control breakpoints
//! - **Auto-retry** — exponential backoff with jitter
//! - **Token counting** — approximate message token estimation
//! - **Multi-provider** — Anthropic, MiniMax, DeepSeek
//!
//! ## Quick start
//!
//! ```rust,no_run
//! use anthropic_rs::{AnthropicClient, ChatMessage};
//!
//! let client = AnthropicClient::new("sk-ant-xxx", "claude-sonnet-4-20250514");
//! let resp = client.messages_create(
//!     &[ChatMessage::user("Hello!")],
//!     Some("You are helpful."), None, 1024,
//! ).unwrap();
//! println!("{}", resp.text);
//! ```

pub mod error;
pub mod types;
pub mod sse;
pub mod api_common;
pub mod request;
pub mod retry;
pub mod tokens;
pub mod cache;

mod client;
mod messages;

#[cfg(feature = "async")]
pub mod async_sse;
#[cfg(feature = "async")]
mod async_client;
#[cfg(feature = "async")]
mod async_messages;

pub use client::AnthropicClient;
#[cfg(feature = "async")]
pub use async_client::AnthropicAsyncClient;
pub use error::{AnthropicError, Result};
pub use types::{
    ChatMessage, ContentBlock, ImageSource, LlmResponse, MessageResponse,
    SimplifiedToolCall, StreamEvent, Tool, ToolCall, ToolChoice, Usage,
};
pub use request::{MessageRequest, ThinkingConfig, Metadata};
pub use retry::RetryConfig;
pub use cache::CacheConfig;
pub use messages::parse_anthropic_stream;