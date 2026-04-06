# LiteRT-LM-Server Documentation

LiteRT-LM-Server is an **OpenAI-compatible HTTP server** intended to wrap **LiteRT-LM** (Google AI Edge) so that existing OpenAI client ecosystems can talk to **local / on-device** models via familiar endpoints.

- Upstream runtime: https://github.com/google-ai-edge/LiteRT-LM
- This repo: https://github.com/lpalbou/LiteRT-LM-Server

## Quick Start (1 command line)

If you have Rust installed, this is the fastest way to get a local server:

```bash
cargo install --locked --git https://github.com/lpalbou/LiteRT-LM-Server \
  && litert-lm-server --listen 127.0.0.1:8080
```

Test:

```bash
curl -s http://127.0.0.1:8080/health
curl -s http://127.0.0.1:8080/v1/models
```

By default the server uses the `mock` backend (no model). For real inference through LiteRT-LM, see `docs/getting-started.md`.

## Read Next

- `docs/getting-started.md`: install/run, auth, streaming, multimodal requests, LiteRT-LM backend setup
- `docs/api.md`: endpoints and request/response shapes (including multimodal input support)
- `docs/architecture.md`: diagrams + how requests map to LiteRT-LM
- `docs/faq.md`: common questions, limitations, security, mobile constraints

