# LiteRT-LM-Server

`LiteRT-LM-Server` is a **universal OpenAI-compatible HTTP wrapper** for **LiteRT-LM**:
https://github.com/google-ai-edge/LiteRT-LM

Goal: make LiteRT-LM usable from the broader “OpenAI client ecosystem” (LangChain/LangGraph/LiteLLM/AbstractCore, etc.) by exposing local, on-device inference through familiar endpoints like:

- `GET /v1/models`
- `POST /v1/chat/completions` (incl. streaming SSE)

## Why Rust?

Rust is a strong fit for a cross-platform “local server + FFI” wrapper:

- **Cross-platform**: macOS/Windows/Linux + iOS/Android toolchains are mature.
- **FFI-friendly**: can call LiteRT-LM via its **C API** (`c/engine.h`) without re-implementing the runtime.
- **Good HTTP stack**: `hyper`/`axum` + SSE streaming is straightforward.
- **Embeddable**: the same code can run as a CLI on desktop, or in-process inside a mobile app (binding to `127.0.0.1`).

## Status

This repo currently provides an OpenAI-compatible server skeleton with a **mock backend**. The LiteRT-LM backend wiring (via C API) is intentionally staged to keep the repo buildable before you set up LiteRT-LM libraries on each platform.

## Run (mock backend)

```bash
cargo run -- --listen 127.0.0.1:8080
```

Test:

```bash
curl http://127.0.0.1:8080/v1/models
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"mock","messages":[{"role":"user","content":"Hello"}]}'
```

## Enable the LiteRT-LM backend (FFI)

This repo is intended to call LiteRT-LM through its **C API** (`c/engine.h`), so the HTTP server stays Rust-only while inference remains in LiteRT-LM.

The `litert` backend is behind a feature flag and requires linking against the LiteRT-LM C API library:

```bash
export LITERT_LM_LIB_DIR=/path/to/litert-lm-c-lib-dir
export LITERT_LM_LIB_NAME=engine        # name without `lib` prefix / extension
export LITERT_LM_LINK_KIND=dylib        # or: static

cargo run --features litert -- \
  --backend litert \
  --model-path /abs/path/to/model.litertlm \
  --litert-backend cpu \
  --listen 127.0.0.1:8080
```

Notes:

- The server currently focuses on the **Conversation** C API (JSON messages in, JSON messages out).
- Tool calling is passed through as OpenAI-style `tools` input; mapping tool-call outputs is implemented for non-stream responses and will be refined.

## Mobile reality check (iOS/Android)

Running a localhost OpenAI-style endpoint on mobile is feasible, but UI/client constraints matter:

- **Pure PWA** (Safari “Add to Home Screen”) generally can’t reliably call `http://127.0.0.1` due to mixed-content/CORS/security policies.
- The practical pattern is a **native wrapper** (React UI in `WKWebView` / Android WebView) that either:
  - calls the runtime via a **JS↔native bridge** (best perf), or
  - talks to a **loopback HTTP server** running in the same app process.

This wrapper server is most valuable when you want to reuse existing OpenAI-compatible tooling while running the model locally.

## Roadmap

- LiteRT-LM backend via the C API (load `.litertlm`, stream tokens, tool-call mapping).
- Session strategy to preserve KV-cache across requests (important for “prompt caching” performance).
- Optional multimodal request mapping (OpenAI content parts → LiteRT-LM image/audio inputs, model-dependent).
