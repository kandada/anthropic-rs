// Copyright (c) 2025 xiefujin <490021684@qq.com>
// Licensed under Apache-2.0, see LICENSE file for full license terms.

//! Shared types for the Anthropic API: chat messages, tool calls, content blocks, etc.

use serde::{Deserialize, Serialize};
use serde_json::Value;

// ── Chat messages ──────────────────────────────────────────────────────────

/// A chat message for the Anthropic Messages API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default)]
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub content_blocks: Option<Vec<ContentBlock>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
}

impl ChatMessage {
    /// Create a user message (text only).
    pub fn user(content: impl Into<String>) -> Self {
        ChatMessage {
            role: "user".into(),
            content: content.into(),
            content_blocks: None,
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
            name: None,
        }
    }

    /// Create a user message with content blocks (for images etc.).
    pub fn user_with_blocks(blocks: Vec<ContentBlock>) -> Self {
        ChatMessage {
            role: "user".into(),
            content: String::new(),
            content_blocks: Some(blocks),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
            name: None,
        }
    }

    /// Create an assistant message (text only).
    pub fn assistant(content: impl Into<String>) -> Self {
        ChatMessage {
            role: "assistant".into(),
            content: content.into(),
            content_blocks: None,
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
            name: None,
        }
    }

    /// Create an assistant message with tool calls.
    pub fn assistant_with_tools(content: impl Into<String>, tool_calls: Vec<ToolCall>) -> Self {
        ChatMessage {
            role: "assistant".into(),
            content: content.into(),
            content_blocks: None,
            tool_calls: Some(tool_calls),
            tool_call_id: None,
            reasoning_content: None,
            name: None,
        }
    }

    /// Create a tool result message.
    pub fn tool_result(tool_use_id: impl Into<String>, content: impl Into<String>) -> Self {
        ChatMessage {
            role: "tool".into(),
            content: content.into(),
            content_blocks: None,
            tool_calls: None,
            tool_call_id: Some(tool_use_id.into()),
            reasoning_content: None,
            name: None,
        }
    }
}

// ── Content blocks (Anthropic format) ───────────────────────────────────────

/// A content block in an Anthropic message. Uses tagged enum for the wire format.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentBlock {
    #[serde(rename = "text")]
    Text {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        citations: Option<Vec<Value>>,
    },
    #[serde(rename = "image")]
    Image {
        source: ImageSource,
    },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
    },
    #[serde(rename = "server_tool_use")]
    ServerToolUse {
        id: String,
        name: String,
        input: Value,
    },
    #[serde(rename = "thinking")]
    Thinking {
        thinking: String,
        signature: String,
    },
    #[serde(rename = "redacted_thinking")]
    RedactedThinking {
        data: String,
    },
    #[serde(rename = "document")]
    Document {
        source: DocumentSource,
        #[serde(skip_serializing_if = "Option::is_none")]
        title: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        context: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        citations: Option<CitationConfig>,
    },
}

/// Image source in a content block.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageSource {
    #[serde(rename = "type")]
    pub source_type: String,
    pub media_type: String,
    pub data: String,
}

/// Document source for document content blocks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentSource {
    #[serde(rename = "type")]
    pub source_type: String,
    pub media_type: Option<String>,
    pub data: Option<String>,
    pub url: Option<String>,
}

/// Citation configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CitationConfig {
    pub enabled: bool,
}

// ── Tool calls ──────────────────────────────────────────────────────────────

/// A tool call from the model (Anthropic format).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub name: String,
    pub input: Value,
}

impl ToolCall {
    /// Convert to a simplified tool call with string arguments.
    pub fn to_simplified(&self) -> SimplifiedToolCall {
        SimplifiedToolCall {
            id: self.id.clone(),
            name: self.name.clone(),
            arguments: serde_json::to_string(&self.input).unwrap_or_default(),
        }
    }
}

// ── Tool definitions ────────────────────────────────────────────────────────

/// A tool definition in Anthropic format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub input_schema: Value,
}

impl Tool {
    pub fn new(name: impl Into<String>, description: impl Into<String>, input_schema: Value) -> Self {
        Tool {
            name: name.into(),
            description: Some(description.into()),
            input_schema,
        }
    }
}

/// Tool choice configuration for Anthropic.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    Auto {
        #[serde(rename = "type")]
        choice_type: String,
    },
    Any {
        #[serde(rename = "type")]
        choice_type: String,
    },
    Tool {
        #[serde(rename = "type")]
        choice_type: String,
        name: String,
    },
}

impl ToolChoice {
    pub fn auto() -> Self {
        ToolChoice::Auto { choice_type: "auto".into() }
    }
    pub fn any() -> Self {
        ToolChoice::Any { choice_type: "any".into() }
    }
    pub fn specific(name: impl Into<String>) -> Self {
        ToolChoice::Tool {
            choice_type: "tool".into(),
            name: name.into(),
        }
    }
}

// ── Response types ──────────────────────────────────────────────────────────

/// A complete message response (non-streaming).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub response_type: String,
    pub role: String,
    #[serde(default)]
    pub content: Vec<ContentBlock>,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequence: Option<String>,
    pub usage: Usage,
}

/// Token usage for Anthropic.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Usage {
    #[serde(default)]
    pub input_tokens: i64,
    #[serde(default)]
    pub output_tokens: i64,
    #[serde(default)]
    pub cache_creation_input_tokens: Option<i64>,
    #[serde(default)]
    pub cache_read_input_tokens: Option<i64>,
    /// Breakdown of output tokens by category (e.g. reasoning).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub output_tokens_details: Option<OutputTokensDetails>,
    /// Server-side tool usage (code execution tool).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub server_tool_use: Option<ServerToolUse>,
}

/// Breakdown of output tokens by category. `output_tokens` remains the
/// inclusive, authoritative total used for billing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputTokensDetails {
    #[serde(default)]
    pub reasoning_tokens: Option<i64>,
    #[serde(default)]
    pub accepted_prediction_tokens: Option<i64>,
    #[serde(default)]
    pub rejected_prediction_tokens: Option<i64>,
}

/// Server-side tool usage counts (e.g. the code execution tool).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerToolUse {
    #[serde(default)]
    pub tool_use_count: Option<i64>,
    #[serde(default)]
    pub input_tokens: Option<i64>,
    #[serde(default)]
    pub cache_creation_input_tokens: Option<i64>,
    #[serde(default)]
    pub cache_read_input_tokens: Option<i64>,
}

/// Response from the `messages/count_tokens` endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CountTokensResponse {
    pub input_tokens: i64,
}

/// Streaming event types ───────────────────────────────────────────────────

/// An Anthropic SSE streaming event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamEvent {
    #[serde(rename = "type")]
    pub event_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<StreamMessageStart>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content_block: Option<ContentBlock>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub delta: Option<StreamDelta>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<ApiError>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamMessageStart {
    pub id: String,
    #[serde(rename = "type")]
    pub msg_type: String,
    pub role: String,
    pub model: String,
}

/// A delta in an Anthropic stream.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamDelta {
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub delta_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,
    /// Carried by the `signature_delta` event that closes a thinking block.
    /// Required to pass the thinking block back in multi-turn conversations.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub partial_json: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequence: Option<String>,
    /// Raw data for `redacted_thinking` blocks (opaque, encrypted).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<String>,
    /// Container info for the code execution (server) tool.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub container: Option<Value>,
}

/// An in-stream error.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiError {
    #[serde(rename = "type")]
    pub error_type: String,
    pub message: String,
}

// ── Simplified response ─────────────────────────────────────────────────────

/// The assembled result of a streamed model call.
#[derive(Debug, Clone, Default)]
pub struct LlmResponse {
    pub text: String,
    pub tool_calls: Vec<SimplifiedToolCall>,
    pub reasoning_content: Option<String>,
    pub finish_reason: Option<String>,
    pub usage: Option<Usage>,
    /// Ordered content blocks in their original stream order (text /
    /// thinking / redacted_thinking / tool_use).
    ///
    /// Thinking blocks carry their `signature`, and `redacted_thinking`
    /// blocks carry their opaque data, so a multi-turn conversation can
    /// pass this vector straight back into `ChatMessage::user_with_blocks`
    /// / assistant message `content_blocks` as Anthropic requires.
    pub content_blocks: Vec<ContentBlock>,
}

impl LlmResponse {
    pub fn is_truncated(&self) -> bool {
        matches!(
            self.finish_reason.as_deref(),
            Some("max_tokens") | Some("connection_closed")
        )
    }
}

/// Simplified tool call for the response envelope.
#[derive(Debug, Clone, PartialEq)]
pub struct SimplifiedToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

impl SimplifiedToolCall {
    pub fn parsed_args(&self) -> Value {
        serde_json::from_str(&self.arguments).unwrap_or_else(|_| serde_json::json!({}))
    }
}