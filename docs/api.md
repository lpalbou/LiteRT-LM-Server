# API

LiteRT-LM-Server aims to be **OpenAI-compatible enough** for common tooling, while staying explicit about what’s supported today.

Base URL examples below assume `http://127.0.0.1:8080`.

## Auth

If you start the server with `--api-key` (or `LITERT_LM_SERVER_API_KEY`), requests must include:

`Authorization: Bearer <key>`

## Endpoints

### `GET /health`

Returns a tiny JSON status.

Response:

```json
{"status":"ok"}
```

### `GET /v1/models`

Returns an OpenAI-style models list.

Response:

```json
{
  "object": "list",
  "data": [
    {"id":"mock","object":"model","created":0,"owned_by":"litert-lm-server"}
  ]
}
```

### `POST /v1/chat/completions`

Supported request fields (today):

- `model` (required): used for routing/echo; `litert` backend exposes its model id via `/v1/models`
- `messages` (required): OpenAI chat messages
- `stream` (optional): when `true`, returns SSE events
- `tools` (optional): OpenAI tool schema array (passed through to LiteRT-LM conversation config)
- `max_tokens` (optional): mapped to LiteRT-LM “max output tokens” (only when using `--backend litert`)
- `temperature`, `top_p` (optional): best-effort mapping to LiteRT-LM sampler params (only when using `--backend litert`)
- `seed` (optional): read from the request body as an extra field (only when using `--backend litert`)
- `top_k` (optional): read from the request body as an extra field (only when using `--backend litert`)

Fields not listed above are currently ignored.

#### Text-only example

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "any-string",
    "messages": [{"role":"user","content":"Hello"}]
  }'
```

#### Multimodal input (images + audio)

LiteRT-LM supports image/audio **inputs** for models that support them. This server accepts:

**A) LiteRT-LM-native parts**

```json
{
  "role": "user",
  "content": [
    {"type":"text","text":"Describe this"},
    {"type":"image","path":"/abs/path/to/image.jpg"},
    {"type":"audio","path":"/abs/path/to/audio.wav"}
  ]
}
```

**B) OpenAI-style `image_url`**

- `data:` URLs are converted into `{ "type":"image","blob":"<base64>" }`
- `file://...` is converted into `{ "type":"image","path":"..." }`
- `http(s)` URLs are rejected (use local paths or data URLs)

```json
{"type":"image_url","image_url":{"url":"data:image/png;base64,<...>"}}
```

**C) OpenAI-style `input_audio`**

```json
{"type":"input_audio","input_audio":{"data":"<base64>"}}
```

#### Tool calling

If your model supports tool calling, LiteRT-LM may return tool calls.

This server returns OpenAI-style `tool_calls` with `function.arguments` encoded as a JSON string.

Your app is responsible for executing tools and sending a `role="tool"` message with the tool output.

## Streaming format (SSE)

When `stream=true`, the server emits:

- an initial chunk with `delta.role="assistant"`
- one or more chunks with `delta.content="..."` (text deltas)
- a final chunk with `finish_reason="stop"`
- a terminal `data: [DONE]`

Example:

```bash
curl -sN http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"any","stream":true,"messages":[{"role":"user","content":"Hello"}]}' \
  | sed -n '1,12p'
```

Limitations (today):

- Streaming tool-call deltas are not fully represented; text is streamed as `delta.content`.

