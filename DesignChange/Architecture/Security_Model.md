# Security Model

Given the immense capability of the LLM execution loops, security boundaries are strictly enforced.

## 1. Authentication (`core.security.auth`)
- Local UI access requires an initialized `AuthUser`.
- API endpoints (`dashboard`) are protected via HMAC-signed session cookies and Bearer Tokens for external webhooks.
- CSRF tokens validate dashboard form submissions.

## 2. Autonomy Governor (`core.autonomy.autonomy_governor`)
- `RiskEvaluator` analyzes the generated `TaskPlanner` intent. Tools in `core.registry.registry` are decorated with `RiskLevel`s.
- Executing an action categorized as `HIGH` risk (e.g. creating/deleting repos via Github, sending Emails, executing destructive OS commands) suspends the thread and requires human-in-the-loop (HITL) approval via the UI or Telegram.

## 3. Sandboxing
- `core.tools.path_utils` restricts LLM file-system manipulation to explicit sandbox directories. Escaping the sandbox (`..` injection) raises fatal pathing exceptions.

## 4. API Keys & Secrets
- Extracted purely from local `.env` files via `core.config.JarvisConfig`. Never logged by `audit.audit_logger`.