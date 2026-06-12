# Deployment Model

Jarvis is designed for localized deployment rather than cloud-hosting to maintain immediate physical device control.

## Containerization
- **Dockerfile**: Exists in the root to allow isolated deployment. However, Docker deployment limits native PyAutoGUI, sounddevice, and system capabilities.
- **Venv Execution**: Standard execution uses Python Virtual Environments (`min_venv`, `jarvis_env`) launched via Powershell `Start.ps1`.

## Local Network Bindings
- FastAPI runs on `0.0.0.0:8000` locally.
- Webhook exposure (for receiving Telegram texts or Twilio events) requires reverse proxies (e.g., ngrok or Cloudflare Tunnels).