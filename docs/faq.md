# FAQ

## Is this a replacement for LangChain / LangGraph / LiteLLM / AbstractCore?

No. This is a **local server wrapper**: it exposes a small subset of OpenAI-compatible endpoints so that higher-level frameworks can talk to **LiteRT-LM** as if it were a “provider”.

You still use orchestration frameworks for:

- agent graphs
- RAG
- tool routing
- observability

## Does it run real models today?

Yes, but only when you enable the `litert` backend and link against the LiteRT-LM C API library.

Out of the box (default `mock` backend), it runs without any model to let you validate integration quickly.

See `docs/getting-started.md` for the LiteRT-LM backend setup.

## Is it fully OpenAI API compatible?

Not yet. It currently focuses on:

- `GET /v1/models`
- `POST /v1/chat/completions` (+ SSE streaming)

Other endpoints (embeddings, responses, audio transcription, files, etc.) are not implemented.

## Does it support vision/audio?

It supports **multimodal inputs** when using the LiteRT-LM backend and a model that supports them:

- image input: `{"type":"image","path":"..."}` or `{"type":"image","blob":"<base64>"}`
- audio input: `{"type":"audio","path":"..."}` or `{"type":"audio","blob":"<base64>"}`

It does not implement image generation or TTS/audio generation.

## Does it execute tools automatically?

No. The server passes `tools` to LiteRT-LM and returns tool calls when the model generates them, but your app/client must:

1) execute tools
2) send a `role="tool"` message with tool output

## Is it safe to expose on a LAN?

By default it binds to `127.0.0.1`. If you bind to `0.0.0.0` you should treat it like any local inference service:

- enable `--api-key`
- consider restricting CORS (currently permissive)
- consider firewall rules

## Does it “prompt cache” across requests?

LiteRT-LM supports stateful sessions and KV-cache reuse when you keep a session/conversation alive.

Today, LiteRT-LM-Server is primarily **stateless per request** (it creates a fresh conversation per call), so you should not expect strong cross-request prompt caching benefits yet.

If you need this, the typical design is a **stateful session mode** keyed by a session id header. That’s on the roadmap.

## Can I use this from a webapp on iPhone (PWA)?

Pure PWAs generally can’t call native on-device inference libraries directly. If your goal is “React UI + local inference” on iOS, the realistic options are:

- run the model in-browser (WebGPU/WebAssembly runtimes), or
- ship a native wrapper app and call LiteRT-LM in-process (best), optionally exposing a loopback HTTP endpoint for compatibility.

## What data does this collect?

LiteRT-LM-Server itself does not include telemetry or analytics. Data exposure depends on:

- where you bind the server (`127.0.0.1` vs LAN)
- whether your tools call the network
- whether you pass remote URLs (remote `http(s)` image URLs are rejected by default)

