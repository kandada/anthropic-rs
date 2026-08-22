// Copyright (c) 2025 xiefujin <490021684@qq.com>
// Licensed under Apache-2.0, see LICENSE file for full license terms.

//! Fine-grained token-counting tests for anthropic-rs.
//!
//! The heuristic here is a rough fallback. The ACCURATE path is the
//! server-computed `messages/count_tokens` endpoint exposed as
//! [`AnthropicClient::count_tokens`] — the same approach the official
//! Anthropic SDK uses (the server counts, no local tokenizer needed).
//! These tests lock in the heuristic's deterministic behavior.

use anthropic_client_rs::*;

#[test]
fn heuristic_is_deterministic() {
    let cases = [
        ("", 0),
        ("Hello world", 3), // 11 ascii chars → ceil(11/4) = 3
        ("你好世界", 2),    // 4 CJK chars × 2pts = 8 → 2
    ];
    for (text, expected) in cases {
        assert_eq!(tokens::count_tokens(text), expected, "{text:?}");
    }
}

#[test]
fn message_count_includes_role_overhead() {
    let msg = ChatMessage::user("Hello world");
    // 4 overhead + content(3)
    assert_eq!(tokens::count_message_tokens(&msg), 7);
}

#[test]
fn system_count_includes_overhead() {
    assert_eq!(tokens::count_system_tokens(""), 0);
    assert_eq!(tokens::count_system_tokens("Hello world"), 7); // 3 + 4
}

#[test]
fn messages_count_is_additive() {
    let msgs = vec![
        ChatMessage::user("Hello"),
        ChatMessage::assistant("world"),
    ];
    let sum = tokens::count_message_tokens(&msgs[0])
        + tokens::count_message_tokens(&msgs[1])
        + 2;
    assert_eq!(tokens::count_messages_tokens(&msgs), sum);
}

#[test]
fn content_blocks_add_to_count() {
    let plain = tokens::count_message_tokens(&ChatMessage::user("hi"));
    let with_blocks = tokens::count_message_tokens(&ChatMessage::user_with_blocks(vec![
        ContentBlock::Text { text: "hi".into(), citations: None },
        ContentBlock::Image {
            source: ImageSource {
                source_type: "base64".into(),
                media_type: "image/png".into(),
                data: "xxxx".into(),
            },
        },
    ]));
    assert!(with_blocks > plain, "image block must add tokens");

    let with_thinking = tokens::count_message_tokens(&{
        let mut m = ChatMessage::assistant("");
        m.content_blocks = Some(vec![ContentBlock::Thinking {
            thinking: "think".into(),
            signature: "sig".into(),
        }]);
        m
    });
    assert!(with_thinking > 0);
}

#[test]
fn count_tokens_endpoint_type_is_present() {
    // The accurate path is the server endpoint; verify the type is exposed.
    let _ = CountTokensResponse { input_tokens: 1 };
}
