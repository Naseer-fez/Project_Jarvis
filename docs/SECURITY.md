# SECURITY

## Dashboard Authentication

The Jarvis dashboard is a local-first web UI. All non-readiness-probe routes
require a bearer token.

### Token Configuration

Set the token via **one** of these methods (env var takes precedence):

```bash
# Option A: environment variable
export JARVIS_DASHBOARD_TOKEN="<strong-random-secret>"

# Option B: CLI flag (sets the env var for the process lifetime)
python main.py --gui --dashboard-token "<strong-random-secret>"
```

Generate a strong token:
```powershell
python -c "import secrets; print(secrets.token_hex(32))"
```

> [!CAUTION]
> The default token is `jarvis`. The server logs a `SECURITY WARNING` at
> startup if the default token is in use. **Never expose the dashboard on a
> non-loopback interface with the default token.**

### Token Usage

| Client type       | How to send the token                                  |
|-------------------|--------------------------------------------------------|
| HTTP (curl/API)   | Header: `X-Dashboard-Token: <token>`                   |
| Browser (via ext) | Header injection extension, or use a reverse proxy     |
| WebSocket         | Query parameter: `ws://host:port/ws?token=<token>`     |

### Unauthenticated Routes

Only `/health` is intentionally unauthenticated. It returns a minimal
`{"ok": true/false, "state": "...", "uptime_seconds": N}` payload with no
internal state.

### Timing-Safe Comparison

`_is_authorized()` uses `hmac.compare_digest` to prevent timing-oracle attacks.

---

## Cloud Fallback Controls

The model router supports Gemini API fallback (see `[models]` in `jarvis.ini`).
Set the API key in `.env`:

```
GEMINI_API_KEY=<your-key>
```

If no API key is configured, cloud models are silently skipped in the
priority chain. Ollama local models are always tried first.

---

## Legacy Controller Fallback

In production (`JARVIS_ENV=production`), the runtime **does not** silently
fall back to the legacy `core.controller` if `core.controller_v2` fails.
This prevents masking real startup regressions.

To deliberately allow the fallback (e.g., during a migration):
```
JARVIS_ALLOW_LEGACY_CONTROLLER=1
```

---

## Secrets Inventory

| Secret                      | Where set              | Purpose                          |
|-----------------------------|------------------------|----------------------------------|
| `JARVIS_DASHBOARD_TOKEN`    | `.env` / CLI           | Dashboard HTTP + WS auth         |
| `GEMINI_API_KEY`            | `.env`                 | Gemini cloud LLM fallback        |
| Telegram bot token          | `.env`                 | Telegram integration             |
| Notion API key              | `.env`                 | Notion integration               |

All secrets are loaded from `.env` via `python-dotenv` on startup.
`.env` is gitignored. Use `.env.example` as the template.

---

## Reporting Security Issues

This is a local-first personal assistant. If you discover a security
vulnerability, open a private GitHub issue or contact the maintainer directly.
