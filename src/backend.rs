use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::BoxStream;

use crate::openai::{ChatCompletionRequest, ModelCard};

#[derive(Debug, Clone)]
pub struct BackendCompletion {
    pub content: Option<String>,
    pub tool_calls: Option<Vec<serde_json::Value>>,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone)]
pub struct BackendChunk {
    pub content_delta: String,
}

#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    #[error("unsupported request: {0}")]
    Unsupported(String),
    #[error("internal backend error: {0}")]
    Internal(String),
}

#[async_trait]
pub trait Backend: Send + Sync {
    fn models(&self) -> Vec<ModelCard>;

    async fn complete(
        &self,
        request: ChatCompletionRequest,
        session_id: Option<&str>,
    ) -> Result<BackendCompletion, BackendError>;

    async fn stream(
        &self,
        request: ChatCompletionRequest,
        session_id: Option<&str>,
    ) -> Result<BoxStream<'static, Result<BackendChunk, BackendError>>, BackendError>;
}

pub type DynBackend = Arc<dyn Backend>;

#[derive(Default)]
pub struct MockBackend;

impl MockBackend {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Backend for MockBackend {
    fn models(&self) -> Vec<ModelCard> {
        vec![ModelCard::new("mock")]
    }

    async fn complete(
        &self,
        request: ChatCompletionRequest,
        session_id: Option<&str>,
    ) -> Result<BackendCompletion, BackendError> {
        let prompt = request
            .messages
            .iter()
            .rev()
            .find_map(|m| m.content_text())
            .unwrap_or("");

        let prefix = session_id
            .map(|s| format!("[session:{s}] "))
            .unwrap_or_default();

        Ok(BackendCompletion {
            content: Some(format!("{prefix}mock backend received: {prompt}")),
            tool_calls: None,
            finish_reason: Some("stop".to_string()),
        })
    }

    async fn stream(
        &self,
        request: ChatCompletionRequest,
        session_id: Option<&str>,
    ) -> Result<BoxStream<'static, Result<BackendChunk, BackendError>>, BackendError> {
        let completion = self.complete(request, session_id).await?;
        let content = completion.content.unwrap_or_default();
        let pieces = content
            .as_bytes()
            .chunks(24)
            .map(|chunk| BackendChunk {
                content_delta: String::from_utf8_lossy(chunk).to_string(),
            })
            .map(Ok)
            .collect::<Vec<_>>();

        Ok(Box::pin(futures::stream::iter(pieces)))
    }
}
