// Copyright (c) 2025 xiefujin <490021684@qq.com>
// Licensed under Apache-2.0, see LICENSE file for full license terms.

//! Request builder for Anthropic Messages API.
//!
//! Provides full control over all Anthropic API parameters.

use serde_json::{json, Value};

use crate::types::{ChatMessage, Tool, ToolChoice};

/// Builder for constructing a Message request body.
///
/// Mirrors all parameters in the [Anthropic Messages API](https://docs.anthropic.com/en/api/messages).
pub struct MessageRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub system: Option<String>,
    pub max_tokens: u64,
    pub stream: bool,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<i64>,
    pub stop_sequences: Option<Vec<String>>,
    pub tools: Option<Vec<Tool>>,
    pub tool_choice: Option<ToolChoice>,
    pub thinking: Option<ThinkingConfig>,
    pub metadata: Option<Metadata>,
}

/// Configuration for extended thinking.
///
/// Mirrors the official Anthropic SDK's `ThinkingConfigParam`, which is a
/// union of `enabled`, `disabled`, and `adaptive`:
/// - `enabled` requires `budget_tokens` (Claude 4.5 and earlier)
/// - `adaptive` needs no budget (Claude 4.6+, recommended going forward)
/// - `disabled` turns thinking off explicitly
///
/// Both `enabled` and `adaptive` accept an optional `display`:
/// - `summarized` — thinking is returned normally (default)
/// - `omitted` — thinking content is redacted but a signature is returned
///   for multi-turn continuity
#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "type")]
pub enum ThinkingConfig {
    #[serde(rename = "enabled")]
    Enabled {
        budget_tokens: u64,
        #[serde(skip_serializing_if = "Option::is_none")]
        display: Option<ThinkingDisplay>,
    },
    #[serde(rename = "disabled")]
    Disabled,
    #[serde(rename = "adaptive")]
    Adaptive {
        #[serde(skip_serializing_if = "Option::is_none")]
        display: Option<ThinkingDisplay>,
    },
}

/// How thinking content appears in the response.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ThinkingDisplay {
    /// Thinking content is returned normally.
    Summarized,
    /// Thinking content is redacted; a signature is returned for
    /// multi-turn continuity.
    Omitted,
}

impl ThinkingConfig {
    /// Fixed-budget extended thinking (requires `budget_tokens`).
    pub fn enabled(budget_tokens: u64) -> Self {
        ThinkingConfig::Enabled { budget_tokens, display: None }
    }

    /// Explicitly disable extended thinking.
    pub fn disabled() -> Self {
        ThinkingConfig::Disabled
    }

    /// Adaptive extended thinking (no budget needed, Claude 4.6+).
    pub fn adaptive() -> Self {
        ThinkingConfig::Adaptive { display: None }
    }

    /// Attach a `display` mode (`summarized` / `omitted`).
    pub fn display(mut self, display: ThinkingDisplay) -> Self {
        match &mut self {
            ThinkingConfig::Enabled { display: d, .. }
            | ThinkingConfig::Adaptive { display: d } => *d = Some(display),
            ThinkingConfig::Disabled => {}
        }
        self
    }
}

/// Metadata for the request (e.g. user_id).
#[derive(Debug, Clone, serde::Serialize)]
pub struct Metadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
}

impl MessageRequest {
    pub fn new(model: impl Into<String>, messages: Vec<ChatMessage>, max_tokens: u64) -> Self {
        MessageRequest {
            model: model.into(),
            messages,
            max_tokens,
            system: None,
            stream: false,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            tools: None,
            tool_choice: None,
            thinking: None,
            metadata: None,
        }
    }

    pub fn system(mut self, v: impl Into<String>) -> Self { self.system = Some(v.into()); self }
    pub fn stream(mut self, v: bool) -> Self { self.stream = v; self }
    pub fn temperature(mut self, v: f64) -> Self { self.temperature = Some(v); self }
    pub fn top_p(mut self, v: f64) -> Self { self.top_p = Some(v); self }
    pub fn top_k(mut self, v: i64) -> Self { self.top_k = Some(v); self }
    pub fn stop_sequences(mut self, v: Vec<String>) -> Self { self.stop_sequences = Some(v); self }
    pub fn tools(mut self, v: Vec<Tool>) -> Self { self.tools = Some(v); self }
    pub fn tool_choice(mut self, v: ToolChoice) -> Self { self.tool_choice = Some(v); self }
    pub fn thinking(mut self, v: ThinkingConfig) -> Self { self.thinking = Some(v); self }
    pub fn metadata(mut self, v: Metadata) -> Self { self.metadata = Some(v); self }

    /// Build the JSON body for this request.
    pub fn build_body(&self) -> Value {
        let msgs = super::api_common::build_anthropic_messages(&self.messages);
        let mut body = json!({
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": msgs,
            "stream": self.stream,
        });

        if let Some(ref sys) = self.system {
            if !sys.is_empty() { body["system"] = json!(sys); }
        }
        if let Some(v) = self.temperature { body["temperature"] = json!(v); }
        if let Some(v) = self.top_p { body["top_p"] = json!(v); }
        if let Some(v) = self.top_k { body["top_k"] = json!(v); }
        if let Some(ref v) = self.stop_sequences { body["stop_sequences"] = json!(v); }
        if let Some(ref v) = self.tools {
            let arr: Vec<Value> = v.iter().map(|t| serde_json::to_value(t).unwrap_or_default()).collect();
            body["tools"] = Value::Array(arr);
        }
        if let Some(ref v) = self.tool_choice {
            body["tool_choice"] = serde_json::to_value(v).unwrap_or_default();
        }
        if let Some(ref v) = self.thinking {
            body["thinking"] = serde_json::to_value(v).unwrap_or_default();
        }
        if let Some(ref v) = self.metadata {
            body["metadata"] = serde_json::to_value(v).unwrap_or_default();
        }

        body
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Tool;
    use serde_json::json;

    fn make_messages() -> Vec<ChatMessage> {
        vec![ChatMessage::user("Hello")]
    }

    #[test]
    fn test_minimal_body() {
        let body = MessageRequest::new("claude-sonnet-4-20250514", make_messages(), 1024).build_body();
        assert_eq!(body["model"], "claude-sonnet-4-20250514");
        assert_eq!(body["max_tokens"], 1024);
        assert_eq!(body["stream"], false);
        assert_eq!(body["messages"].as_array().unwrap().len(), 1);
        assert!(body.get("system").is_none());
    }

    #[test]
    fn test_full_body() {
        let tools = vec![Tool::new("fn", "desc", json!({"type":"object","properties":{}}))];
        let body = MessageRequest::new("claude-sonnet-4-20250514", make_messages(), 2048)
            .system("You are helpful.")
            .temperature(0.7)
            .top_p(0.9)
            .top_k(40)
            .stop_sequences(vec!["END".into()])
            .tools(tools)
            .tool_choice(ToolChoice::auto())
            .thinking(ThinkingConfig::enabled(4096))
            .metadata(Metadata { user_id: Some("user-1".into()) })
            .build_body();

        assert_eq!(body["system"], "You are helpful.");
        assert_eq!(body["temperature"], 0.7);
        assert_eq!(body["top_p"], 0.9);
        assert_eq!(body["top_k"], 40);
        assert_eq!(body["stop_sequences"].as_array().unwrap()[0], "END");
        assert!(body["tools"].is_array());
        assert_eq!(body["thinking"]["type"], "enabled");
        assert_eq!(body["thinking"]["budget_tokens"], 4096);
        assert_eq!(body["metadata"]["user_id"], "user-1");
    }

    #[test]
    fn test_system_empty_omitted() {
        let body = MessageRequest::new("claude-sonnet-4-20250514", make_messages(), 1024)
            .system("")
            .build_body();
        assert!(body.get("system").is_none());
    }

    #[test]
    fn test_body_serializable() {
        let body = MessageRequest::new("claude-sonnet-4-20250514", make_messages(), 1024)
            .temperature(0.5)
            .tools(vec![Tool::new("f", "d", json!({"type":"object","properties":{}}))])
            .build_body();
        let s = serde_json::to_string(&body).unwrap();
        assert!(s.contains("claude-sonnet-4-20250514"));
        assert!(s.contains("tools"));
    }

    #[test]
    fn test_stream_body() {
        let body = MessageRequest::new("claude-sonnet-4-20250514", make_messages(), 1024)
            .stream(true)
            .build_body();
        assert_eq!(body["stream"], true);
    }

    #[test]
    fn test_thinking_adaptive_no_budget() {
        let body = MessageRequest::new("claude-sonnet-4-20250514", make_messages(), 1024)
            .thinking(ThinkingConfig::adaptive())
            .build_body();
        assert_eq!(body["thinking"]["type"], "adaptive");
        assert!(body["thinking"].get("budget_tokens").is_none());
    }

    #[test]
    fn test_thinking_disabled() {
        let body = MessageRequest::new("claude-sonnet-4-20250514", make_messages(), 1024)
            .thinking(ThinkingConfig::disabled())
            .build_body();
        assert_eq!(body["thinking"]["type"], "disabled");
    }

    #[test]
    fn test_thinking_display_omitted() {
        let body = MessageRequest::new("claude-sonnet-4-20250514", make_messages(), 1024)
            .thinking(ThinkingConfig::adaptive().display(ThinkingDisplay::Omitted))
            .build_body();
        assert_eq!(body["thinking"]["type"], "adaptive");
        assert_eq!(body["thinking"]["display"], "omitted");
    }
}