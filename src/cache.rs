// Copyright (c) 2025 xiefujin <490021684@qq.com>
// Licensed under Apache-2.0, see LICENSE file for full license terms.

//! Prompt caching support for Anthropic API.
//!
//! Adds `cache_control` breakpoints to content blocks for
//! [Anthropic Prompt Caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching).

use serde_json::Value;

/// Add cache_control breakpoints to a system prompt.
pub fn cache_system(body: &mut Value) {
    if let Some(sys) = body.get("system") {
        if sys.is_string() {
            let arr = serde_json::json!([
                {"type": "text", "text": sys.as_str().unwrap(), "cache_control": {"type": "ephemeral"}}
            ]);
            body["system"] = arr;
        }
    }
}

/// Add cache_control to the last tool in the tools array.
pub fn cache_last_tool(body: &mut Value) {
    if let Some(tools) = body.get_mut("tools").and_then(|t| t.as_array_mut()) {
        if let Some(last) = tools.last_mut() {
            last["cache_control"] = serde_json::json!({"type": "ephemeral"});
        }
    }
}

/// Add cache_control to the last content block in a message.
pub fn cache_last_message_block(message: &mut Value) {
    if let Some(content) = message.get_mut("content").and_then(|c| c.as_array_mut()) {
        if let Some(last) = content.last_mut() {
            last["cache_control"] = serde_json::json!({"type": "ephemeral"});
        }
    }
}

/// Configuration for cache breakpoints.
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Whether to cache the system prompt.
    pub cache_system: bool,
    /// Whether to cache the last tool definition.
    pub cache_tools: bool,
    /// Number of messages from the end to cache.
    pub cache_last_messages: usize,
}

impl Default for CacheConfig {
    fn default() -> Self {
        CacheConfig {
            cache_system: false,
            cache_tools: false,
            cache_last_messages: 0,
        }
    }
}

impl CacheConfig {
    /// Enable caching for system, tools, and last N messages.
    pub fn standard() -> Self {
        CacheConfig {
            cache_system: true,
            cache_tools: true,
            cache_last_messages: 2,
        }
    }

    /// Apply cache breakpoints to a request body.
    pub fn apply(&self, body: &mut Value) {
        if self.cache_system {
            cache_system(body);
        }
        if self.cache_tools {
            cache_last_tool(body);
        }
        if self.cache_last_messages > 0 {
            if let Some(msgs) = body.get_mut("messages").and_then(|m| m.as_array_mut()) {
                let start = msgs.len().saturating_sub(self.cache_last_messages);
                for msg in msgs.iter_mut().skip(start) {
                    cache_last_message_block(msg);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_system() {
        let mut body = serde_json::json!({"system": "You are helpful."});
        cache_system(&mut body);
        let sys = &body["system"];
        assert!(sys.is_array());
        assert_eq!(sys[0]["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn test_cache_last_tool() {
        let mut body = serde_json::json!({
            "tools": [
                {"name": "a", "input_schema": {}},
                {"name": "b", "input_schema": {}}
            ]
        });
        cache_last_tool(&mut body);
        assert_eq!(body["tools"][1]["cache_control"]["type"], "ephemeral");
    }
}