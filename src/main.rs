mod backend;
#[cfg(feature = "litert")]
mod litert_backend;
#[cfg(feature = "litert")]
mod litert_ffi;
mod openai;

use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use clap::Parser;
use futures::StreamExt;
use futures::stream::BoxStream;
use serde_json::json;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;
use tracing::{error, info};
use tracing_subscriber::EnvFilter;
use uuid::Uuid;

use crate::backend::{BackendError, DynBackend, MockBackend};
#[cfg(feature = "litert")]
use crate::litert_backend::{LitertBackend, LitertBackendConfig};
use crate::openai::{
    ChatCompletionChunk, ChatCompletionChunkChoice, ChatCompletionRequest, ChatCompletionResponse,
    ChatMessage, ModelsListResponse, OpenAiError, OpenAiErrorResponse, Usage,
};

#[derive(Debug, Parser)]
#[command(name = "litert-lm-server", version)]
struct Args {
    /// Address to bind to (use `127.0.0.1` for loopback-only).
    #[arg(
        long,
        default_value = "127.0.0.1:8080",
        env = "LITERT_LM_SERVER_LISTEN"
    )]
    listen: SocketAddr,

    /// Backend implementation: `mock` or `litert`.
    #[arg(long, default_value = "mock", env = "LITERT_LM_SERVER_BACKEND")]
    backend: String,

    /// Path to a `.litertlm`/`.tflite`/`.task` model file (required for `--backend litert`).
    #[arg(long, env = "LITERT_LM_SERVER_MODEL_PATH")]
    model_path: Option<std::path::PathBuf>,

    /// LiteRT-LM backend string (e.g. `cpu`, `gpu`, `npu`) for `--backend litert`.
    #[arg(long, default_value = "cpu", env = "LITERT_LM_SERVER_LITERT_BACKEND")]
    litert_backend: String,

    /// Vision backend string for multimodal models (optional).
    #[arg(long, env = "LITERT_LM_SERVER_VISION_BACKEND")]
    vision_backend: Option<String>,

    /// Audio backend string for multimodal models (optional).
    #[arg(long, env = "LITERT_LM_SERVER_AUDIO_BACKEND")]
    audio_backend: Option<String>,

    /// Cache directory for GPU program cache / weight cache.
    #[arg(long, env = "LITERT_LM_SERVER_CACHE_DIR")]
    cache_dir: Option<std::path::PathBuf>,

    /// Maximum tokens for the model context.
    #[arg(long, env = "LITERT_LM_SERVER_MAX_NUM_TOKENS")]
    max_num_tokens: Option<i32>,

    /// Enable constrained decoding (LiteRT-LM supports token-level constraints in C++).
    #[arg(
        long,
        env = "LITERT_LM_SERVER_ENABLE_CONSTRAINED_DECODING",
        default_value_t = false
    )]
    enable_constrained_decoding: bool,

    /// LiteRT-LM C API min log level (0=INFO,1=WARNING,2=ERROR,3=FATAL).
    #[arg(long, env = "LITERT_LM_MIN_LOG_LEVEL")]
    litert_min_log_level: Option<i32>,

    /// OpenAI-style API key to require (expects `Authorization: Bearer <key>`).
    #[arg(long, env = "LITERT_LM_SERVER_API_KEY")]
    api_key: Option<String>,
}

#[derive(Clone)]
struct AppState {
    backend: DynBackend,
    api_key: Option<Arc<str>>,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let args = Args::parse();

    let backend: DynBackend = match args.backend.as_str() {
        "mock" => Arc::new(MockBackend::new()),
        "litert" => {
            #[cfg(feature = "litert")]
            {
                let model_path = args.model_path.unwrap_or_else(|| {
                    eprintln!("--model-path is required when --backend=litert");
                    std::process::exit(2);
                });
                Arc::new(
                    LitertBackend::new(LitertBackendConfig {
                        model_path,
                        backend: args.litert_backend.clone(),
                        vision_backend: args.vision_backend.clone(),
                        audio_backend: args.audio_backend.clone(),
                        cache_dir: args.cache_dir.clone(),
                        max_num_tokens: args.max_num_tokens,
                        enable_constrained_decoding: args.enable_constrained_decoding,
                        min_log_level: args.litert_min_log_level,
                        model_id: None,
                    })
                    .unwrap_or_else(|e| {
                        eprintln!("failed to initialize LiteRT-LM backend: {e}");
                        std::process::exit(2);
                    }),
                )
            }
            #[cfg(not(feature = "litert"))]
            {
                eprintln!(
                    "backend=litert requires building with `--features litert` (see README for linking setup)"
                );
                std::process::exit(2);
            }
        }
        other => {
            eprintln!("unknown backend: {other} (expected: mock|litert)");
            std::process::exit(2);
        }
    };

    let state = AppState {
        backend,
        api_key: args.api_key.map(Arc::from),
    };

    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_headers(Any)
                .allow_methods(Any),
        )
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    info!("listening on {}", args.listen);
    let listener = tokio::net::TcpListener::bind(args.listen)
        .await
        .expect("bind failed");
    axum::serve(listener, app).await.expect("server failed");
}

async fn health() -> impl IntoResponse {
    Json(json!({"status": "ok"}))
}

async fn list_models(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if let Err(resp) = ensure_authorized(&headers, &state) {
        return resp;
    }

    let data = state.backend.models();
    Json(ModelsListResponse {
        object: "list".to_string(),
        data,
    })
    .into_response()
}

async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(request): Json<ChatCompletionRequest>,
) -> Response {
    if let Err(resp) = ensure_authorized(&headers, &state) {
        return resp;
    }

    let user_session = request.user.clone();
    let requested_model = request.model.clone();
    let session_id = headers
        .get("x-litert-session-id")
        .and_then(|v| v.to_str().ok())
        .or(user_session.as_deref());

    if request.stream {
        return chat_completions_stream(state, request, session_id).await;
    }

    match state.backend.complete(request, session_id).await {
        Ok(result) => {
            let created = unix_time_seconds();
            let id = format!("chatcmpl-{}", Uuid::new_v4());

            Json(ChatCompletionResponse {
                id,
                object: "chat.completion".to_string(),
                created,
                model: requested_model,
                choices: vec![crate::openai::ChatCompletionChoice {
                    index: 0,
                    message: ChatMessage {
                        role: "assistant".to_string(),
                        content: result.content.map(|c| json!(c)),
                        name: None,
                        tool_call_id: None,
                        tool_calls: result.tool_calls,
                    },
                    finish_reason: result.finish_reason.or(Some("stop".to_string())),
                }],
                usage: Some(Usage {
                    prompt_tokens: 0,
                    completion_tokens: 0,
                    total_tokens: 0,
                }),
            })
            .into_response()
        }
        Err(err) => backend_error(err),
    }
}

async fn chat_completions_stream(
    state: AppState,
    request: ChatCompletionRequest,
    session_id: Option<&str>,
) -> Response {
    let created = unix_time_seconds();
    let id = format!("chatcmpl-{}", Uuid::new_v4());
    let model = request.model.clone();

    let stream: Result<BoxStream<'static, Result<Event, Infallible>>, BackendError> =
        state.backend.stream(request, session_id).await.map(|s| {
            let id_for_chunks = id.clone();
            let model_for_chunks = model.clone();
            let chunks = s.map(move |chunk_result| match chunk_result {
                Ok(chunk) => {
                    let mut delta = std::collections::HashMap::new();
                    delta.insert("content".to_string(), json!(chunk.content_delta));
                    let payload = ChatCompletionChunk {
                        id: id_for_chunks.clone(),
                        object: "chat.completion.chunk".to_string(),
                        created,
                        model: model_for_chunks.clone(),
                        choices: vec![ChatCompletionChunkChoice {
                            index: 0,
                            delta,
                            finish_reason: None,
                        }],
                    };
                    let json = serde_json::to_string(&payload).unwrap_or_else(|e| {
                        error!("failed to serialize chunk: {e}");
                        "{}".to_string()
                    });
                    Ok(Event::default().data(json))
                }
                Err(e) => {
                    error!("backend stream error: {e}");
                    Ok(Event::default().data(
                        serde_json::to_string(&OpenAiErrorResponse {
                            error: OpenAiError {
                                message: e.to_string(),
                                r#type: "server_error".to_string(),
                                param: None,
                                code: None,
                            },
                        })
                        .unwrap_or_else(|_| "{}".to_string()),
                    ))
                }
            });

            // Emit a role chunk first for clients that expect it.
            let role = {
                let mut delta = std::collections::HashMap::new();
                delta.insert("role".to_string(), json!("assistant"));
                let payload = ChatCompletionChunk {
                    id: id.clone(),
                    object: "chat.completion.chunk".to_string(),
                    created,
                    model: model.clone(),
                    choices: vec![ChatCompletionChunkChoice {
                        index: 0,
                        delta,
                        finish_reason: None,
                    }],
                };
                let json = serde_json::to_string(&payload).unwrap_or_else(|_| "{}".to_string());
                futures::stream::once(async move { Ok(Event::default().data(json)) })
            };

            let finish = {
                let delta = std::collections::HashMap::new();
                let payload = ChatCompletionChunk {
                    id: id.clone(),
                    object: "chat.completion.chunk".to_string(),
                    created,
                    model: model.clone(),
                    choices: vec![ChatCompletionChunkChoice {
                        index: 0,
                        delta,
                        finish_reason: Some("stop".to_string()),
                    }],
                };
                let json = serde_json::to_string(&payload).unwrap_or_else(|_| "{}".to_string());
                futures::stream::once(async move { Ok(Event::default().data(json)) })
            };

            let done = futures::stream::once(async { Ok(Event::default().data("[DONE]")) });

            Box::pin(role.chain(chunks).chain(finish).chain(done))
                as BoxStream<'static, Result<Event, Infallible>>
        });

    match stream {
        Ok(sse_stream) => Sse::new(sse_stream)
            .keep_alive(KeepAlive::new().interval(std::time::Duration::from_secs(15)))
            .into_response(),
        Err(err) => backend_error(err),
    }
}

fn ensure_authorized(headers: &HeaderMap, state: &AppState) -> Result<(), Response> {
    let Some(expected) = state.api_key.as_deref() else {
        return Ok(());
    };

    let Some(authz) = headers.get(axum::http::header::AUTHORIZATION) else {
        return Err(openai_error(
            StatusCode::UNAUTHORIZED,
            "missing Authorization header",
        ));
    };

    let Ok(authz) = authz.to_str() else {
        return Err(openai_error(
            StatusCode::UNAUTHORIZED,
            "invalid Authorization header",
        ));
    };

    let expected = format!("Bearer {expected}");
    if authz != expected {
        return Err(openai_error(StatusCode::UNAUTHORIZED, "invalid API key"));
    }

    Ok(())
}

fn backend_error(err: BackendError) -> Response {
    match err {
        BackendError::Unsupported(msg) => openai_error(StatusCode::BAD_REQUEST, &msg),
        BackendError::Internal(msg) => openai_error(StatusCode::INTERNAL_SERVER_ERROR, &msg),
    }
}

fn openai_error(status: StatusCode, message: &str) -> Response {
    let body = OpenAiErrorResponse {
        error: OpenAiError {
            message: message.to_string(),
            r#type: "invalid_request_error".to_string(),
            param: None,
            code: None,
        },
    };
    (status, Json(body)).into_response()
}

fn unix_time_seconds() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}
