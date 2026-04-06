# Getting Started

## 0) Install prerequisites

- Rust toolchain (stable): https://www.rust-lang.org/tools/install

## 1) Run the server (mock backend) in one command

```bash
cargo install --locked --git https://github.com/lpalbou/LiteRT-LM-Server \
  && litert-lm-server --listen 127.0.0.1:8080
```

Verify:

```bash
curl -s http://127.0.0.1:8080/health
curl -s http://127.0.0.1:8080/v1/models
```

## 2) Make a chat request

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"mock","messages":[{"role":"user","content":"Hello"}]}' | jq .
```

Streaming:

```bash
curl -sN http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"mock","stream":true,"messages":[{"role":"user","content":"Hello"}]}' \
  | sed -n '1,12p'
```

## 3) Add simple auth (recommended)

Run with an API key requirement:

```bash
LITERT_LM_SERVER_API_KEY='dev-key' litert-lm-server --listen 127.0.0.1:8080
```

Then call with:

```bash
curl -s http://127.0.0.1:8080/v1/models \
  -H 'Authorization: Bearer dev-key'
```

## 4) Enable the LiteRT-LM backend (real inference)

The LiteRT-LM backend uses LiteRT-LM’s **C API** (`c/engine.h`). In practice, this means:

1) Build the LiteRT-LM C API library (static `.a` is easiest), then
2) link it into this server with `--features litert`.

### 4.1 Build LiteRT-LM C API library (Bazel)

From a checkout of `google-ai-edge/LiteRT-LM`:

```bash
# Full engine (may include more deps)
bazel build -c opt //c:engine

# CPU-only engine (usually easier to link)
bazel build -c opt //c:engine_cpu
```

To find the produced archive path:

```bash
bazel cquery //c:engine_cpu --output=files
```

Example output (macOS/arm64):

`bazel-out/darwin_arm64-opt/bin/c/libengine_cpu.a`

### 4.2 Run LiteRT-LM-Server against LiteRT-LM (one command line)

```bash
LITERT_LM_LIB_DIR="/abs/path/to/LiteRT-LM/bazel-out/<platform>-opt/bin/c" \
LITERT_LM_LIB_NAME="engine_cpu" \
LITERT_LM_LINK_KIND="static" \
cargo run --release --features litert -- \
  --backend litert \
  --model-path /abs/path/to/model.litertlm \
  --litert-backend cpu \
  --listen 127.0.0.1:8080
```

Notes:

- Use `--litert-backend gpu` or `--litert-backend npu` only if your LiteRT-LM build + platform supports it.
- Use `--cache-dir /some/writable/dir` to enable LiteRT program/weight caches (2nd load faster on some backends).
- Multimodal inputs require a **multimodal model** and may need `--vision-backend` / `--audio-backend`.

## 5) Multimodal requests (images + audio)

LiteRT-LM supports image/audio **inputs** for models that support them. This server accepts multimodal input in two compatible styles:

### 5.1 LiteRT-LM-native parts (recommended)

```json
{
  "role": "user",
  "content": [
    {"type": "text", "text": "Describe this image"},
    {"type": "image", "path": "/abs/path/to/image.jpg"}
  ]
}
```

Audio is analogous:

```json
{"type":"audio","path":"/abs/path/to/audio.wav"}
```

You can also pass bytes as base64:

```json
{"type":"image","blob":"<base64 bytes>"}
```

### 5.2 OpenAI-style `image_url` parts (data URL or local file URL)

```json
{
  "role": "user",
  "content": [
    {"type": "text", "text": "Describe this"},
    {
      "type": "image_url",
      "image_url": { "url": "data:image/jpeg;base64,<...>" }
    }
  ]
}
```

Remote `http(s)` image URLs are intentionally rejected (use local paths or `data:` URLs).

### 5.3 OpenAI-style `input_audio` (base64)

```json
{"type":"input_audio","input_audio":{"data":"<base64>"}}
```

## 6) Tool calling

You can pass OpenAI-style `tools` in the request; LiteRT-LM will attempt to generate tool calls for models that support tool calling.

This server does **not** automatically execute tools; your client/app is responsible for:

1) reading tool calls in the response,
2) executing them, and
3) sending a `role="tool"` message with the tool result.

## Troubleshooting

- If `--backend litert` fails to link: make sure you set `LITERT_LM_LINK_KIND=static` when linking `libengine*.a`, and that your `LITERT_LM_LIB_DIR` points to the directory containing it.
- On Apple platforms, linking C++ code requires libc++; `build.rs` adds `-lc++` automatically when `--features litert` is enabled.

