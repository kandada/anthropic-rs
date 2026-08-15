// Copyright (c) 2025 xiefujin <490021684@qq.com>
// Licensed under Apache-2.0, see LICENSE file for full license terms.

//! Async SSE reader for Anthropic streams.
//!
//! Accumulates bytes from a `reqwest` streaming response and yields
//! `data:` payloads one at a time. Handles UTF-8 BOM stripping.

use futures::StreamExt;
use crate::error::AnthropicError;

pub struct AsyncSseStream<S> {
    stream: S,
    buffer: Vec<u8>,
    done: bool,
    first_read: bool,
}

impl<S> AsyncSseStream<S>
where
    S: futures::Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Unpin,
{
    pub fn new(stream: S) -> Self {
        AsyncSseStream { stream, buffer: Vec::new(), done: false, first_read: true }
    }

    pub async fn next_data(&mut self) -> Result<Option<String>, AnthropicError> {
        if self.done { return Ok(None); }
        loop {
            if let Some(pos) = self.buffer.iter().position(|&b| b == b'\n') {
                let line_bytes = self.buffer[..pos].to_vec();
                self.buffer.drain(..=pos);
                let line = String::from_utf8_lossy(&line_bytes).into_owned();
                let line = line.trim_end_matches('\r').to_string();
                if self.first_read {
                    self.first_read = false;
                    if line.starts_with('\u{FEFF}') {
                        let stripped = line[3..].to_string();
                        if stripped.trim_end_matches(['\r', '\n']).is_empty() { continue; }
                        if let Some(payload) = Self::extract_payload(&stripped) {
                            if payload == "[DONE]" { self.done = true; return Ok(None); }
                            return Ok(Some(payload));
                        }
                        continue;
                    }
                    if line.trim_end_matches(['\r', '\n']).is_empty() { continue; }
                }
                let trimmed = line.trim_end_matches(['\r', '\n']);
                if trimmed.is_empty() { continue; }
                if let Some(payload) = Self::extract_payload(trimmed) {
                    if payload == "[DONE]" { self.done = true; return Ok(None); }
                    return Ok(Some(payload));
                }
                continue;
            }
            match self.stream.next().await {
                Some(Ok(chunk)) => { self.buffer.extend_from_slice(&chunk); continue; }
                Some(Err(e)) => return Err(AnthropicError::Network(format!("SSE stream error: {e}"))),
                None => {
                    if !self.buffer.is_empty() {
                        let line_bytes = std::mem::take(&mut self.buffer);
                        let line = String::from_utf8_lossy(&line_bytes).into_owned();
                        let line = line.trim_end_matches(['\r', '\n']).to_string();
                        if let Some(payload) = Self::extract_payload(&line) {
                            if payload == "[DONE]" { self.done = true; return Ok(None); }
                            return Ok(Some(payload));
                        }
                    }
                    self.done = true;
                    return Ok(None);
                }
            }
        }
    }

    fn extract_payload(line: &str) -> Option<String> {
        line.strip_prefix("data:").map(|rest| rest.strip_prefix(' ').unwrap_or(rest).to_string())
    }
}