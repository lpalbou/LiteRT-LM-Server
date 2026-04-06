# LiteRT-LM-Server

`LiteRT-LM-Server` is a **universal OpenAI-compatible HTTP wrapper** for **LiteRT-LM**:
https://github.com/google-ai-edge/LiteRT-LM

Goal: make LiteRT-LM usable from the broader “OpenAI client ecosystem” (LangChain/LangGraph/LiteLLM/AbstractCore, etc.) by exposing local, on-device inference through familiar endpoints like:

- `GET /v1/models`
- `POST /v1/chat/completions` (incl. streaming SSE)

## Docs

- `docs/README.md`
- `docs/getting-started.md`
- `docs/api.md`
- `docs/architecture.md`
- `docs/faq.md`

## Why Rust?

Rust is a strong fit for a cross-platform “local server + FFI” wrapper:

- **Cross-platform**: macOS/Windows/Linux + iOS/Android toolchains are mature.
- **FFI-friendly**: can call LiteRT-LM via its **C API** (`c/engine.h`) without re-implementing the runtime.
- **Good HTTP stack**: `hyper`/`axum` + SSE streaming is straightforward.
- **Embeddable**: the same code can run as a CLI on desktop, or in-process inside a mobile app (binding to `127.0.0.1`).

## Status

This repo provides:

- an OpenAI-compatible server skeleton with a **mock backend** (no model), and
- a LiteRT-LM backend behind `--features litert` (links to LiteRT-LM C API).

## Run (mock backend)

```bash
cargo install --locked --git https://github.com/lpalbou/LiteRT-LM-Server \
  && litert-lm-server --listen 127.0.0.1:8080
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
export LITERT_LM_LINK_KIND=static       # recommended when linking Bazel-built `libengine*.a`

cargo run --features litert -- \
  --backend litert \
  --model-path /abs/path/to/model.litertlm \
  --litert-backend cpu \
  --listen 127.0.0.1:8080
```

Notes:

- The server currently focuses on the **Conversation** C API (JSON messages in, JSON messages out).
- Multimodal request mapping (image/audio inputs) is supported for the LiteRT-LM backend.

## Mobile reality check (iOS/Android)

Running a localhost OpenAI-style endpoint on mobile is feasible, but UI/client constraints matter:

- **Pure PWA** (Safari “Add to Home Screen”) generally can’t reliably call `http://127.0.0.1` due to mixed-content/CORS/security policies.
- The practical pattern is a **native wrapper** (React UI in `WKWebView` / Android WebView) that either:
  - calls the runtime via a **JS↔native bridge** (best perf), or
  - talks to a **loopback HTTP server** running in the same app process.

This wrapper server is most valuable when you want to reuse existing OpenAI-compatible tooling while running the model locally.

## Roadmap

- Session strategy to preserve KV-cache across requests (important for “prompt caching” performance).
- Broader OpenAI surface area (additional endpoints, richer streaming semantics).
