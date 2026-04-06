#![cfg(feature = "litert")]

use std::ffi::{CStr, CString};
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use futures::stream::BoxStream;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use uuid::Uuid;

use crate::backend::{Backend, BackendChunk, BackendCompletion, BackendError};
use crate::litert_ffi as ffi;
use crate::openai::{ChatCompletionRequest, ChatMessage, ModelCard};

pub struct LitertBackendConfig {
    pub model_path: PathBuf,
    pub backend: String,
    pub vision_backend: Option<String>,
    pub audio_backend: Option<String>,
    pub cache_dir: Option<PathBuf>,
    pub max_num_tokens: Option<i32>,
    pub enable_constrained_decoding: bool,
    pub min_log_level: Option<i32>,
    pub model_id: Option<String>,
}

pub struct LitertBackend {
    engine: EngineHandle,
    model_id: String,
    enable_constrained_decoding: bool,
}

impl LitertBackend {
    pub fn new(config: LitertBackendConfig) -> Result<Self, BackendError> {
        unsafe {
            if let Some(level) = config.min_log_level {
                ffi::litert_lm_set_min_log_level(level);
            }
        }

        let model_id = config.model_id.unwrap_or_else(|| {
            config
                .model_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("litert-model")
                .to_string()
        });

        let settings = EngineSettingsHandle::create(
            &config.model_path,
            &config.backend,
            config.vision_backend.as_deref(),
            config.audio_backend.as_deref(),
        )?;

        if let Some(max_num_tokens) = config.max_num_tokens {
            unsafe {
                ffi::litert_lm_engine_settings_set_max_num_tokens(
                    settings.ptr.as_ptr(),
                    max_num_tokens,
                );
            }
        }
        if let Some(cache_dir) = config.cache_dir.as_deref() {
            let cache_dir = CString::new(cache_dir.to_string_lossy().as_bytes())
                .map_err(|e| BackendError::Internal(e.to_string()))?;
            unsafe {
                ffi::litert_lm_engine_settings_set_cache_dir(
                    settings.ptr.as_ptr(),
                    cache_dir.as_ptr(),
                );
            }
        }

        let engine = EngineHandle::create(&settings)?;

        Ok(Self {
            engine,
            model_id,
            enable_constrained_decoding: config.enable_constrained_decoding,
        })
    }

    fn split_history_and_last(
        request: &ChatCompletionRequest,
    ) -> Result<(&[ChatMessage], &ChatMessage), BackendError> {
        if request.messages.is_empty() {
            return Err(BackendError::Unsupported(
                "messages must be non-empty".to_string(),
            ));
        }
        let last_index = request.messages.len() - 1;
        Ok((
            &request.messages[..last_index],
            &request.messages[last_index],
        ))
    }

    fn to_litert_message_json(message: &ChatMessage) -> Result<serde_json::Value, BackendError> {
        use serde_json::json;

        let role = message.role.as_str();
        match role {
            "system" | "user" | "assistant" => {
                let content = match message.content.as_ref() {
                    None | Some(serde_json::Value::Null) => vec![],
                    Some(serde_json::Value::String(s)) => vec![json!({"type":"text","text": s})],
                    Some(serde_json::Value::Object(obj)) => {
                        // Allow a single content part object (e.g. {"type":"text","text":"..."}).
                        vec![serde_json::Value::Object(obj.clone())]
                    }
                    Some(serde_json::Value::Array(parts)) => parts
                        .iter()
                        .filter_map(|p| {
                            let ty = p.get("type")?.as_str()?;
                            if ty == "text" {
                                let text = p.get("text")?.as_str()?;
                                Some(json!({"type":"text","text": text}))
                            } else {
                                None
                            }
                        })
                        .collect(),
                    Some(other) => {
                        return Err(BackendError::Unsupported(format!(
                            "unsupported message content for role={role}: {other}"
                        )));
                    }
                };

                Ok(json!({
                  "role": role,
                  "content": content,
                }))
            }
            "tool" => {
                // LiteRT-LM expects tool response content to be a JSON object.
                let content = match message.content.as_ref() {
                    Some(serde_json::Value::Object(obj)) => serde_json::Value::Object(obj.clone()),
                    Some(serde_json::Value::String(s)) => {
                        serde_json::from_str::<serde_json::Value>(s)
                            .ok()
                            .and_then(|v| v.as_object().cloned().map(serde_json::Value::Object))
                            .unwrap_or_else(|| serde_json::json!({ "result": s }))
                    }
                    None | Some(serde_json::Value::Null) => serde_json::json!({}),
                    Some(other) => {
                        return Err(BackendError::Unsupported(format!(
                            "unsupported tool message content: {other}"
                        )));
                    }
                };

                Ok(json!({
                  "role": "tool",
                  "content": content,
                }))
            }
            other => Err(BackendError::Unsupported(format!(
                "unsupported role: {other}"
            ))),
        }
    }

    fn extract_text_and_tool_calls(
        litert_message: &serde_json::Value,
    ) -> (Option<String>, Option<Vec<serde_json::Value>>) {
        let content_text = match litert_message.get("content") {
            Some(serde_json::Value::String(s)) => Some(s.clone()),
            Some(serde_json::Value::Array(parts)) => {
                let mut out = String::new();
                for part in parts {
                    if part.get("type").and_then(|v| v.as_str()) == Some("text") {
                        if let Some(t) = part.get("text").and_then(|v| v.as_str()) {
                            out.push_str(t);
                        }
                    }
                }
                if out.is_empty() { None } else { Some(out) }
            }
            _ => None,
        };

        let tool_calls = litert_message
            .get("tool_calls")
            .and_then(|v| v.as_array())
            .filter(|arr| !arr.is_empty())
            .map(|arr| {
                arr.iter()
                    .map(|tc| {
                        let name = tc
                            .get("function")
                            .and_then(|f| f.get("name"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let args_val = tc
                            .get("function")
                            .and_then(|f| f.get("arguments"))
                            .cloned()
                            .unwrap_or_else(|| serde_json::json!({}));
                        let args_str = if let Some(s) = args_val.as_str() {
                            s.to_string()
                        } else {
                            serde_json::to_string(&args_val).unwrap_or_else(|_| "{}".to_string())
                        };
                        serde_json::json!({
                          "id": format!("call_{}", Uuid::new_v4()),
                          "type": "function",
                          "function": {
                            "name": name,
                            "arguments": args_str
                          }
                        })
                    })
                    .collect::<Vec<_>>()
            });

        (content_text, tool_calls)
    }

    fn tools_json(request: &ChatCompletionRequest) -> Option<String> {
        request
            .tools
            .as_ref()
            .and_then(|t| serde_json::to_string(t).ok())
    }

    fn history_json(history: &[ChatMessage]) -> Option<String> {
        if history.is_empty() {
            return None;
        }
        let mut out = Vec::with_capacity(history.len());
        for msg in history {
            if let Ok(j) = Self::to_litert_message_json(msg) {
                out.push(j);
            }
        }
        serde_json::to_string(&out).ok()
    }

    fn create_ephemeral_conversation(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<ConversationHandle, BackendError> {
        let (history, _last) = Self::split_history_and_last(request)?;

        let tools_json = Self::tools_json(request);
        let messages_json = Self::history_json(history);

        let tools_c = tools_json.as_deref().and_then(|s| CString::new(s).ok());
        let messages_c = messages_json.as_deref().and_then(|s| CString::new(s).ok());

        let config_ptr = unsafe {
            ffi::litert_lm_conversation_config_create(
                self.engine.ptr.as_ptr(),
                std::ptr::null(),
                std::ptr::null(),
                tools_c
                    .as_ref()
                    .map(|s| s.as_ptr())
                    .unwrap_or(std::ptr::null()),
                messages_c
                    .as_ref()
                    .map(|s| s.as_ptr())
                    .unwrap_or(std::ptr::null()),
                self.enable_constrained_decoding,
            )
        };

        let mut config = ConversationConfigHandle::new(config_ptr)?;
        let conv_ptr = unsafe {
            ffi::litert_lm_conversation_create(self.engine.ptr.as_ptr(), config.ptr.as_ptr())
        };
        // Conversation copies config, so we can drop it immediately.
        config.delete();
        ConversationHandle::new(conv_ptr)
    }
}

#[async_trait]
impl Backend for LitertBackend {
    fn models(&self) -> Vec<ModelCard> {
        vec![ModelCard::new(self.model_id.clone())]
    }

    async fn complete(
        &self,
        request: ChatCompletionRequest,
        _session_id: Option<&str>,
    ) -> Result<BackendCompletion, BackendError> {
        let (_history, last) = Self::split_history_and_last(&request)?;
        let last_json = Self::to_litert_message_json(last)?;
        let message_json = CString::new(
            serde_json::to_string(&last_json).map_err(|e| BackendError::Internal(e.to_string()))?,
        )
        .map_err(|e| BackendError::Internal(e.to_string()))?;

        let conversation = self.create_ephemeral_conversation(&request)?;

        let resp_ptr = unsafe {
            ffi::litert_lm_conversation_send_message(
                conversation.ptr.as_ptr(),
                message_json.as_ptr(),
                std::ptr::null(),
            )
        };
        let resp = JsonResponseHandle::new(resp_ptr)?;
        let json_str = unsafe {
            let ptr = ffi::litert_lm_json_response_get_string(resp.ptr.as_ptr());
            if ptr.is_null() {
                return Err(BackendError::Internal("null json response".to_string()));
            }
            CStr::from_ptr(ptr).to_string_lossy().to_string()
        };

        let parsed: serde_json::Value = serde_json::from_str(&json_str)
            .unwrap_or_else(|_| serde_json::json!({ "content": json_str }));

        let (content, tool_calls) = Self::extract_text_and_tool_calls(&parsed);
        let finish_reason = if tool_calls.is_some() {
            Some("tool_calls".to_string())
        } else {
            Some("stop".to_string())
        };

        Ok(BackendCompletion {
            content,
            tool_calls,
            finish_reason,
        })
    }

    async fn stream(
        &self,
        request: ChatCompletionRequest,
        _session_id: Option<&str>,
    ) -> Result<BoxStream<'static, Result<BackendChunk, BackendError>>, BackendError> {
        let (_history, last) = Self::split_history_and_last(&request)?;
        let last_json = Self::to_litert_message_json(last)?;
        let message_json = CString::new(
            serde_json::to_string(&last_json).map_err(|e| BackendError::Internal(e.to_string()))?,
        )
        .map_err(|e| BackendError::Internal(e.to_string()))?;

        let conversation = self.create_ephemeral_conversation(&request)?;

        let (tx, rx) = mpsc::channel::<Result<BackendChunk, BackendError>>(64);
        let callback_data = Arc::new(CallbackData { sender: tx });
        let raw = Arc::into_raw(callback_data.clone()) as *mut std::ffi::c_void;

        let rc = unsafe {
            ffi::litert_lm_conversation_send_message_stream(
                conversation.ptr.as_ptr(),
                message_json.as_ptr(),
                std::ptr::null(),
                Some(stream_callback),
                raw,
            )
        };
        if rc != 0 {
            // Reclaim the leaked Arc.
            unsafe { drop(Arc::from_raw(raw as *const CallbackData)) };
            return Err(BackendError::Internal(format!(
                "failed to start LiteRT-LM stream (code {rc})"
            )));
        }

        let stream = ReceiverStream::new(rx).map(move |item| {
            // Keep the conversation alive for the duration of the stream.
            let _keep = &conversation;
            item
        });

        Ok(Box::pin(stream))
    }
}

struct CallbackData {
    sender: mpsc::Sender<Result<BackendChunk, BackendError>>,
}

unsafe extern "C" fn stream_callback(
    callback_data: *mut std::ffi::c_void,
    chunk: *const std::os::raw::c_char,
    is_final: bool,
    error_msg: *const std::os::raw::c_char,
) {
    if callback_data.is_null() {
        return;
    }

    let arc = unsafe { Arc::from_raw(callback_data as *const CallbackData) };

    if !error_msg.is_null() {
        let msg = unsafe { CStr::from_ptr(error_msg).to_string_lossy().to_string() };
        let _ = arc.sender.blocking_send(Err(BackendError::Internal(msg)));
    } else if !chunk.is_null() {
        let s = unsafe { CStr::from_ptr(chunk).to_string_lossy().to_string() };
        // Stream callback returns JSON messages; extract any text content parts.
        let text = serde_json::from_str::<serde_json::Value>(&s)
            .ok()
            .and_then(|v| match v.get("content") {
                Some(serde_json::Value::Array(parts)) => {
                    let mut out = String::new();
                    for part in parts {
                        if part.get("type").and_then(|v| v.as_str()) == Some("text") {
                            if let Some(t) = part.get("text").and_then(|v| v.as_str()) {
                                out.push_str(t);
                            }
                        }
                    }
                    if out.is_empty() { None } else { Some(out) }
                }
                Some(serde_json::Value::String(s)) if !s.is_empty() => Some(s.clone()),
                _ => None,
            })
            .unwrap_or_else(|| s.clone());

        if !text.is_empty() {
            let _ = arc.sender.blocking_send(Ok(BackendChunk {
                content_delta: text,
            }));
        }
    }

    if is_final {
        // Drop the C-owned reference.
        return;
    }

    // Keep the Arc alive for the next callback invocation.
    let _ = Arc::into_raw(arc);
}

struct EngineSettingsHandle {
    ptr: NonNull<ffi::LiteRtLmEngineSettings>,
}

impl EngineSettingsHandle {
    fn create(
        model_path: &Path,
        backend: &str,
        vision_backend: Option<&str>,
        audio_backend: Option<&str>,
    ) -> Result<Self, BackendError> {
        let model_path = CString::new(model_path.to_string_lossy().as_bytes())
            .map_err(|e| BackendError::Internal(e.to_string()))?;
        let backend = CString::new(backend).map_err(|e| BackendError::Internal(e.to_string()))?;
        let vision_backend = vision_backend.and_then(|s| CString::new(s).ok());
        let audio_backend = audio_backend.and_then(|s| CString::new(s).ok());

        let ptr = unsafe {
            ffi::litert_lm_engine_settings_create(
                model_path.as_ptr(),
                backend.as_ptr(),
                vision_backend
                    .as_ref()
                    .map(|s| s.as_ptr())
                    .unwrap_or(std::ptr::null()),
                audio_backend
                    .as_ref()
                    .map(|s| s.as_ptr())
                    .unwrap_or(std::ptr::null()),
            )
        };

        let ptr = NonNull::new(ptr).ok_or_else(|| {
            BackendError::Internal("litert_lm_engine_settings_create returned null".to_string())
        })?;
        Ok(Self { ptr })
    }
}

impl Drop for EngineSettingsHandle {
    fn drop(&mut self) {
        unsafe { ffi::litert_lm_engine_settings_delete(self.ptr.as_ptr()) }
    }
}

struct EngineHandle {
    ptr: NonNull<ffi::LiteRtLmEngine>,
}

unsafe impl Send for EngineHandle {}
unsafe impl Sync for EngineHandle {}

impl EngineHandle {
    fn create(settings: &EngineSettingsHandle) -> Result<Self, BackendError> {
        let ptr = unsafe { ffi::litert_lm_engine_create(settings.ptr.as_ptr()) };
        let ptr = NonNull::new(ptr).ok_or_else(|| {
            BackendError::Internal("litert_lm_engine_create returned null".to_string())
        })?;
        Ok(Self { ptr })
    }
}

impl Drop for EngineHandle {
    fn drop(&mut self) {
        unsafe { ffi::litert_lm_engine_delete(self.ptr.as_ptr()) }
    }
}

struct ConversationConfigHandle {
    ptr: NonNull<ffi::LiteRtLmConversationConfig>,
    deleted: bool,
}

impl ConversationConfigHandle {
    fn new(ptr: *mut ffi::LiteRtLmConversationConfig) -> Result<Self, BackendError> {
        let ptr = NonNull::new(ptr).ok_or_else(|| {
            BackendError::Internal("litert_lm_conversation_config_create returned null".to_string())
        })?;
        Ok(Self {
            ptr,
            deleted: false,
        })
    }

    fn delete(&mut self) {
        if !self.deleted {
            unsafe { ffi::litert_lm_conversation_config_delete(self.ptr.as_ptr()) };
            self.deleted = true;
        }
    }
}

impl Drop for ConversationConfigHandle {
    fn drop(&mut self) {
        self.delete();
    }
}

struct ConversationHandle {
    ptr: NonNull<ffi::LiteRtLmConversation>,
}

unsafe impl Send for ConversationHandle {}
unsafe impl Sync for ConversationHandle {}

impl ConversationHandle {
    fn new(ptr: *mut ffi::LiteRtLmConversation) -> Result<Self, BackendError> {
        let ptr = NonNull::new(ptr).ok_or_else(|| {
            BackendError::Internal("litert_lm_conversation_create returned null".to_string())
        })?;
        Ok(Self { ptr })
    }
}

impl Drop for ConversationHandle {
    fn drop(&mut self) {
        unsafe { ffi::litert_lm_conversation_delete(self.ptr.as_ptr()) }
    }
}

struct JsonResponseHandle {
    ptr: NonNull<ffi::LiteRtLmJsonResponse>,
}

impl JsonResponseHandle {
    fn new(ptr: *mut ffi::LiteRtLmJsonResponse) -> Result<Self, BackendError> {
        let ptr = NonNull::new(ptr).ok_or_else(|| {
            BackendError::Internal("litert_lm_conversation_send_message returned null".to_string())
        })?;
        Ok(Self { ptr })
    }
}

impl Drop for JsonResponseHandle {
    fn drop(&mut self) {
        unsafe { ffi::litert_lm_json_response_delete(self.ptr.as_ptr()) }
    }
}
