# Architecture

This doc describes how LiteRT-LM-Server maps OpenAI-style HTTP requests onto LiteRT-LM’s local inference APIs.

## 1) High-level view

```mermaid
flowchart TB
  Client["Client\n(OpenAI SDKs, LangChain/LangGraph,\nLiteLLM, AbstractCore, curl, etc.)"]
  Server["LiteRT-LM-Server\n(axum/hyper)"]

  subgraph Backends["Backends"]
    Mock["Mock backend\n(no model)"]
    LiteRT["LiteRT-LM backend\n(C API FFI)"]
  end

  subgraph LiteRTLM["LiteRT-LM (upstream)"]
    CAPI["C API\n(c/engine.h)"]
    Conv["Conversation\n(messages/templates/tools)"]
    Exec["Executor\n(LiteRT delegates)"]
    HW["CPU / GPU / NPU"]
  end

  Client -->|"HTTP /v1/*"| Server
  Server --> Mock
  Server --> LiteRT
  LiteRT --> CAPI --> Conv --> Exec --> HW
```

## 2) Request lifecycle: `/v1/chat/completions`

### Non-streaming

```mermaid
sequenceDiagram
  participant C as Client
  participant S as Server
  participant B as Backend
  participant L as LiteRT-LM (Conversation)

  C->>S: POST /v1/chat/completions (messages, tools, ...)
  S->>B: complete(request)
  B->>L: Create ConversationConfig (preface=history, tools)
  B->>L: SendMessage(last_message)
  L-->>B: JSON Message (assistant content + optional tool_calls)
  B-->>S: content/tool_calls
  S-->>C: OpenAI ChatCompletionResponse
```

### Streaming (SSE)

```mermaid
sequenceDiagram
  participant C as Client
  participant S as Server
  participant B as Backend
  participant L as LiteRT-LM (Conversation)

  C->>S: POST /v1/chat/completions (stream=true)
  S->>B: stream(request)
  B->>L: SendMessageAsync(...) + callback
  loop chunks
    L-->>B: callback(chunk JSON)
    B-->>S: chunk text delta
    S-->>C: SSE data: {chat.completion.chunk ...}
  end
  S-->>C: SSE data: [DONE]
```

## 3) Multimodal mapping

LiteRT-LM expects message parts like:

- `{ "type": "text",  "text": "..." }`
- `{ "type": "image", "path": "..." }` or `{ "type": "image", "blob": "<base64>" }`
- `{ "type": "audio", "path": "..." }` or `{ "type": "audio", "blob": "<base64>" }`

This server accepts either:

1) those native parts directly, or
2) OpenAI-style parts like `image_url` (mapped into `image` parts).

```mermaid
flowchart LR
  OA["OpenAI request message\ncontent parts"] --> Map["Part mapper\n(OpenAI -> LiteRT-LM)"]
  Map --> LM["LiteRT-LM message JSON\n(role + content parts)"]
  LM --> Conv["LiteRT-LM Conversation\n(model-specific processors)"]
```

## 4) FFI boundary

LiteRT-LM-Server stays Rust-only at the HTTP layer. Inference is delegated to LiteRT-LM via its C API.

```mermaid
flowchart TB
  Rust["Rust server\n(axum)"]
  FFI["FFI boundary\n(src/litert_ffi.rs)"]
  CAPI["LiteRT-LM C API\n(c/engine.h)"]
  CPP["LiteRT-LM C++ core\n(runtime/*)"]
  LiteRT["LiteRT delegates\n(CPU/GPU/NPU)"]

  Rust --> FFI --> CAPI --> CPP --> LiteRT
```

