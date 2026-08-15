// Copyright (c) 2025 xiefujin <490021684@qq.com>
// Licensed under Apache-2.0, see LICENSE file for full license terms.

//! Token counting utility for Anthropic models.
//!
//! Uses character-based heuristics for approximate token counts.
//! For production use, consider an Anthropic-compatible tokenizer.

use crate::types::{ChatMessage, ContentBlock};

fn count_tokens(text: &str) -> u64 {
    if text.is_empty() { return 0; }
    let mut tokens = 0;
    for ch in text.chars() {
        tokens += if ch.is_ascii() { 1 } else { 3 };
    }
    ((tokens as f64) / 4.0).ceil() as u64
}

/// Approximate token count for a message.
pub fn count_message_tokens(msg: &ChatMessage) -> u64 {
    let mut total = 4;
    total += count_tokens(&msg.content);
    if let Some(ref blocks) = msg.content_blocks {
        for block in blocks {
            match block {
                ContentBlock::Text { text, .. } => total += count_tokens(text),
                ContentBlock::Image { .. } => total += 100,
                ContentBlock::ToolUse { name, input, .. } => {
                    total += count_tokens(name);
                    total += count_tokens(&serde_json::to_string(input).unwrap_or_default());
                    total += 10;
                }
                ContentBlock::ToolResult { content, .. } => total += count_tokens(content) + 5,
                ContentBlock::Thinking { thinking, .. } => total += count_tokens(thinking),
                ContentBlock::Document { .. } => total += 200,
                _ => total += 10,
            }
        }
    }
    if let Some(ref tcs) = msg.tool_calls {
        for tc in tcs {
            total += count_tokens(&tc.name);
            total += count_tokens(&serde_json::to_string(&tc.input).unwrap_or_default());
            total += 10;
        }
    }
    if let Some(ref rc) = msg.reasoning_content {
        total += count_tokens(rc);
    }
    total
}

/// Approximate total token count for a list of messages.
pub fn count_messages_tokens(messages: &[ChatMessage]) -> u64 {
    let mut total = 0;
    for msg in messages {
        total += count_message_tokens(msg) + 1;
    }
    total
}

/// Count tokens for a system prompt.
pub fn count_system_tokens(system: &str) -> u64 {
    if system.is_empty() { 0 } else { count_tokens(system) + 4 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_empty() { assert_eq!(count_tokens(""), 0); }

    #[test]
    fn test_count_english() {
        let n = count_tokens("Hello world");
        assert!(n >= 2, "got {n}");
    }

    #[test]
    fn test_count_message() {
        let msg = ChatMessage::user("Hello world");
        let n = count_message_tokens(&msg);
        assert!(n >= 5, "got {n}");
    }

    #[test]
    fn test_count_with_image() {
        let msg = ChatMessage::user_with_blocks(vec![
            ContentBlock::Text { text: "hi".into(), citations: None },
            ContentBlock::Image { source: crate::types::ImageSource {
                source_type: "base64".into(), media_type: "image/png".into(), data: "xxx".into(),
            }},
        ]);
        let n = count_message_tokens(&msg);
        assert!(n > 100, "should include image cost, got {n}");
    }
}