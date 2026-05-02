# Jarvis .env API Keys Guide

This guide explains which values Jarvis can read from `.env`, which ones are required, and where to get each API key or credential.

You do not need every key. Jarvis is local-first: you can run it with Ollama and add cloud providers or integrations only when you want those features.

## Before You Start

Use these files as your reference:

- `.env` - your real local secrets. Do not commit this file.
- `.env.example` - starter template for core cloud and integration keys.
- `config/settings.env.template` - optional integration/provider variables.

Important safety rules:

- Never paste real keys into GitHub, Discord, chats, screenshots, issues, or docs.
- If a real key was exposed anywhere, revoke or rotate it in that provider's dashboard.
- Prefer least-privilege tokens when a provider supports scopes or repository limits.
- Add billing limits or usage alerts for paid cloud providers.
- Keep `.env` in the project root: `D:\AI\Jarvis\.env`.

## Minimum Setup

For a practical local setup, start with only this:

```env
JARVIS_ENV=development
JARVIS_DASHBOARD_TOKEN=replace_with_a_long_random_secret
OLLAMA_BASE_URL=http://localhost:11434
CLOUD_LLM_FALLBACK_ENABLED=false
```

Then install and run Ollama:

```powershell
ollama serve
ollama pull deepseek-r1:8b
ollama pull mistral:7b
ollama pull llava
```

Notes:

- `JARVIS_DASHBOARD_TOKEN` is not from a website. You create it yourself.
- Ollama does not require an API key.
- Current code reads `OLLAMA_BASE_URL`; `.env.example` also contains older `OLLAMA_ENDPOINT`.

To create a strong dashboard token in PowerShell:

```powershell
[Convert]::ToBase64String((1..32 | ForEach-Object { Get-Random -Minimum 0 -Maximum 256 }))
```

## Key Map

| Feature | Variables | Required? | Where to get it |
| --- | --- | --- | --- |
| Local Ollama | `OLLAMA_BASE_URL`, optional `OLLAMA_MODEL` | No API key | Install from `https://ollama.com/` |
| Dashboard auth | `JARVIS_DASHBOARD_TOKEN` | Recommended | Generate locally |
| Production auth | `JARVIS_SECRET_KEY`, `JARVIS_ADMIN_USER`, `JARVIS_ADMIN_PASSWORD` | Production only | Generate locally |
| Gemini fallback | `GEMINI_API_KEY` | Optional | Google AI Studio |
| Groq fallback | `GROQ_API_KEY` | Optional | GroqCloud Console |
| OpenAI fallback | `OPENAI_API_KEY` | Optional | OpenAI Platform |
| Anthropic fallback | `ANTHROPIC_API_KEY` | Optional | Anthropic Console |
| Telegram | `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` | Optional | Telegram BotFather and Bot API |
| Gmail and Google Calendar | `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `GOOGLE_REFRESH_TOKEN` | Optional | Google Cloud Console and OAuth Playground |
| Notion | `NOTION_API_KEY` | Optional | Notion integrations page |
| Spotify | `SPOTIFY_CLIENT_ID`, `SPOTIFY_CLIENT_SECRET`, `SPOTIFY_REFRESH_TOKEN` | Optional | Spotify Developer Dashboard |
| Home Assistant | `HOME_ASSISTANT_URL`, `HOME_ASSISTANT_TOKEN` | Optional | Home Assistant profile page |
| GitHub | `GITHUB_TOKEN`, optional `GITHUB_DEFAULT_REPO` | Optional | GitHub Developer Settings |
| IMAP/SMTP email | `EMAIL_ADDRESS`, `EMAIL_PASSWORD`, `SMTP_HOST`, `SMTP_PORT`, `IMAP_HOST` | Optional | Your email provider |
| Twilio WhatsApp | `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_WHATSAPP_FROM` | Optional | Twilio Console |
| Wake word | `PORCUPINE_ACCESS_KEY` | Optional | Picovoice Console |
| Tavily web search | `TAVILY_API_KEY` | Optional | Tavily dashboard |
| OpenWeatherMap weather | `WEATHER_API_KEY` | Optional | OpenWeather dashboard |
| Local calendar file | `CALENDAR_ICS_PATH` | Optional | No key needed |

## Local Secrets You Generate Yourself

### Dashboard Token

Used by the dashboard for protected POST and WebSocket actions.

```env
JARVIS_DASHBOARD_TOKEN=your_long_random_secret
```

Generate it locally:

```powershell
[Convert]::ToBase64String((1..32 | ForEach-Object { Get-Random -Minimum 0 -Maximum 256 }))
```

### Production Secrets

Only needed if you run with `JARVIS_ENV=production`.

```env
JARVIS_ENV=production
JARVIS_SECRET_KEY=your_long_random_secret
JARVIS_ADMIN_USER=your_admin_username
JARVIS_ADMIN_PASSWORD=your_long_admin_password
```

Production guardrails in the code expect `JARVIS_SECRET_KEY` to be strong and non-default. The admin password should be at least 12 characters.

## Cloud LLM Providers

Set `CLOUD_LLM_FALLBACK_ENABLED=true` only if you want Jarvis to call cloud LLMs when local Ollama routing is unavailable or insufficient.

```env
CLOUD_LLM_FALLBACK_ENABLED=true
GEMINI_API_KEY=
GROQ_API_KEY=
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
```

### Gemini

Use this first if you want one cloud fallback key.

Official pages:

- `https://aistudio.google.com/apikey`
- `https://ai.google.dev/tutorials/setup`

Steps:

1. Open Google AI Studio.
2. Sign in.
3. Open the API Keys page.
4. Create a key in a Google Cloud project.
5. Copy it once and add it to `.env`.

```env
GEMINI_API_KEY=your_gemini_key
```

### Groq

Official page:

- `https://console.groq.com/keys`

Steps:

1. Open GroqCloud Console.
2. Go to API Keys.
3. Create a key.
4. Add it to `.env`.

```env
GROQ_API_KEY=your_groq_key
```

### OpenAI

Official page:

- `https://platform.openai.com/api-keys`

Steps:

1. Open the OpenAI Platform API keys page.
2. Create a new secret key.
3. Copy it immediately. Secret keys are shown only once.
4. Add it to `.env`.

```env
OPENAI_API_KEY=your_openai_key
```

### Anthropic

Official pages:

- `https://console.anthropic.com/settings/keys`
- `https://docs.anthropic.com/en/docs/get-started`

Steps:

1. Open Anthropic Console.
2. Go to API Keys.
3. Create a key in the right workspace.
4. Add it to `.env`.

```env
ANTHROPIC_API_KEY=your_anthropic_key
```

## Telegram

Used by `send_telegram` and `get_updates`.

```env
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
```

Official pages:

- `https://core.telegram.org/bots/features#botfather`
- `https://core.telegram.org/bots/api#getupdates`

Steps:

1. Open Telegram.
2. Search for `@BotFather`.
3. Send `/newbot`.
4. Follow the prompts and copy the bot token.
5. Send a message to your new bot from the chat you want Jarvis to use.
6. Open this URL in a browser, replacing `<TOKEN>` with the bot token:

```text
https://api.telegram.org/bot<TOKEN>/getUpdates
```

7. In the JSON response, find `chat.id`.
8. Put the token and chat ID in `.env`.

Tips:

- Personal chat IDs are usually positive numbers.
- Group chat IDs are often negative numbers.
- If the bot needs to read group messages, check BotFather privacy settings.

## Google Calendar and Gmail

Used by Google Calendar and Gmail integrations. Both active clients use the same OAuth credentials.

```env
GOOGLE_CLIENT_ID=
GOOGLE_CLIENT_SECRET=
GOOGLE_REFRESH_TOKEN=
```

Official pages:

- Google Cloud credentials: `https://console.cloud.google.com/apis/credentials`
- OAuth clients help: `https://support.google.com/cloud/answer/6158849`
- OAuth Playground: `https://developers.google.com/oauthplayground/`
- Gmail scopes: `https://developers.google.com/workspace/gmail/api/auth/scopes`

Steps:

1. Open Google Cloud Console.
2. Create or select a project.
3. Enable these APIs:
   - Gmail API
   - Google Calendar API
4. Configure the OAuth consent screen.
5. Add your Google account as a test user if the app is in testing mode.
6. Create an OAuth client.
7. For the easiest OAuth Playground flow, choose `Web application`.
8. Add this authorized redirect URI:

```text
https://developers.google.com/oauthplayground
```

9. Copy the Client ID and Client Secret.
10. Open OAuth Playground.
11. Click the settings gear.
12. Enable `Use your own OAuth credentials`.
13. Paste your Client ID and Client Secret.
14. Select the scopes Jarvis needs:

```text
https://www.googleapis.com/auth/calendar.events
https://www.googleapis.com/auth/gmail.readonly
https://www.googleapis.com/auth/gmail.send
https://www.googleapis.com/auth/gmail.modify
```

15. Authorize APIs.
16. Exchange the authorization code for tokens.
17. Copy the refresh token.
18. Add all three values to `.env`.

If no refresh token appears, revoke the test app from your Google Account permissions and repeat the OAuth flow. Google often returns the refresh token only on the first consent.

## Notion

Used by Notion page, database, and block tools.

```env
NOTION_API_KEY=
```

Official pages:

- `https://www.notion.com/my-integrations`
- `https://developers.notion.com/guides/get-started/internal-integrations`

Steps:

1. Open Notion's integrations page.
2. Create an internal integration.
3. Copy the internal integration token.
4. Share each page or database Jarvis should access with that integration.
5. Add the token to `.env`.

Notes:

- New Notion tokens may start with `ntn_`; older guides often show `secret_`.
- `NOTION_DATABASE_ID` appears in some local `.env` setups, but the current active client accepts database IDs as tool arguments.

## Spotify

Used by playback, search, currently-playing, and playlist tools.

```env
SPOTIFY_CLIENT_ID=
SPOTIFY_CLIENT_SECRET=
SPOTIFY_REFRESH_TOKEN=
```

Optional local helper value:

```env
SPOTIFY_REDIRECT_URI=http://127.0.0.1:8888/callback
```

Official pages:

- Developer Dashboard: `https://developer.spotify.com/dashboard`
- Authorization Code Flow: `https://developer.spotify.com/documentation/web-api/tutorials/code-flow`
- Refreshing tokens: `https://developer.spotify.com/documentation/web-api/tutorials/refreshing-tokens`

Steps:

1. Open Spotify Developer Dashboard.
2. Create an app.
3. Copy the Client ID and Client Secret.
4. Add a redirect URI, for example:

```text
http://127.0.0.1:8888/callback
```

5. Run the Authorization Code flow once.
6. Request these scopes:

```text
user-read-playback-state
user-modify-playback-state
user-read-currently-playing
playlist-modify-public
playlist-modify-private
```

7. Exchange the authorization code for an access token and refresh token.
8. Copy the refresh token into `.env`.

Shortcut for the authorization URL:

```text
https://accounts.spotify.com/authorize?response_type=code&client_id=YOUR_CLIENT_ID&scope=user-read-playback-state%20user-modify-playback-state%20user-read-currently-playing%20playlist-modify-public%20playlist-modify-private&redirect_uri=http%3A%2F%2F127.0.0.1%3A8888%2Fcallback&state=jarvis
```

After approval, Spotify redirects to the redirect URI with `?code=...`. Copy that code and exchange it at:

```text
https://accounts.spotify.com/api/token
```

The current Jarvis Spotify client uses the refresh token with Client ID and Client Secret.

## Home Assistant

Used by smart-home entity and service tools.

```env
HOME_ASSISTANT_URL=http://homeassistant.local:8123
HOME_ASSISTANT_TOKEN=
```

Official pages:

- REST API: `https://developers.home-assistant.io/docs/api/rest`
- Auth API: `https://developers.home-assistant.io/docs/auth_api/`

Steps:

1. Open your Home Assistant frontend.
2. Click your user profile.
3. Scroll to Long-Lived Access Tokens.
4. Create a token named `Jarvis`.
5. Copy it immediately.
6. Add your Home Assistant URL and token to `.env`.

Example:

```env
HOME_ASSISTANT_URL=http://homeassistant.local:8123
HOME_ASSISTANT_TOKEN=your_long_lived_access_token
```

## GitHub

Used by GitHub issue, PR, gist, and code-search tools.

```env
GITHUB_TOKEN=
GITHUB_DEFAULT_REPO=owner/repo
```

Official page:

- `https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token`

Recommended steps:

1. Open GitHub.
2. Go to Settings.
3. Open Developer settings.
4. Open Personal access tokens.
5. Prefer Fine-grained tokens when possible.
6. Limit the token to only the repositories Jarvis should touch.
7. Choose an expiration date.
8. Grant only the permissions you need.

Suggested fine-grained permissions:

- Metadata: read
- Contents: read, for repository inspection and code search
- Issues: read/write, if Jarvis should create or close issues
- Pull requests: read, if Jarvis should list PRs or read diffs

Gist note:

- `create_gist` may require a classic token with the `gist` scope. If you do not need gist creation, skip that permission.

## IMAP/SMTP Email

Used by the generic email integration, separate from the Gmail API integration above.

```env
EMAIL_ADDRESS=
EMAIL_PASSWORD=
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
IMAP_HOST=imap.gmail.com
```

Official Gmail references:

- Gmail IMAP/SMTP: `https://developers.google.com/gmail/imap/imap-smtp`
- Google App Passwords: `https://support.google.com/accounts/answer/2461835`

Steps for Gmail:

1. Turn on 2-Step Verification.
2. Create an App Password for Mail.
3. Use your Gmail address as `EMAIL_ADDRESS`.
4. Use the app password as `EMAIL_PASSWORD`.
5. Keep the Gmail hosts from the example above.

For other providers, use that provider's IMAP host, SMTP host, port, and app-password flow.

## Twilio WhatsApp

Used by the WhatsApp integration.

```env
TWILIO_ACCOUNT_SID=
TWILIO_AUTH_TOKEN=
TWILIO_WHATSAPP_FROM=whatsapp:+14155238886
```

Official pages:

- Twilio Console: `https://console.twilio.com/`
- WhatsApp quickstart: `https://www.twilio.com/docs/whatsapp/quickstart`
- WhatsApp sandbox: `https://www.twilio.com/docs/conversations/use-twilio-sandbox-for-whatsapp`

Steps:

1. Open Twilio Console.
2. Copy your Account SID.
3. Copy your Auth Token.
4. Open the WhatsApp Sandbox or approved WhatsApp sender setup.
5. Copy the sender number in `whatsapp:+E164_NUMBER` format.
6. Add the values to `.env`.

For the sandbox, `TWILIO_WHATSAPP_FROM` is often:

```env
TWILIO_WHATSAPP_FROM=whatsapp:+14155238886
```

## Picovoice Porcupine

Used by wake-word voice mode.

```env
PORCUPINE_ACCESS_KEY=
```

Official page:

- `https://picovoice.ai/docs/porcupine/`

Steps:

1. Open Picovoice Console.
2. Sign in.
3. Copy your AccessKey from the console.
4. Add it to `.env`.

## Tavily Web Search

Used only if you want Tavily-backed web search.

```env
WEB_SEARCH_ENABLED=true
WEB_SEARCH_PROVIDER=auto
TAVILY_API_KEY=
```

Official pages:

- `https://docs.tavily.com/api-reference/introduction`
- `https://app.tavily.com/`

Steps:

1. Open Tavily.
2. Create an API key.
3. Add it to `.env`.
4. Keep `WEB_SEARCH_PROVIDER=auto` if you want Jarvis to use Tavily when the key is present and fall back otherwise.

## OpenWeatherMap

Used by the legacy weather integration, which calls OpenWeatherMap's current weather endpoint.

```env
WEATHER_API_KEY=
```

Official page:

- `https://openweathermap.org/api`

Steps:

1. Create or sign in to an OpenWeather account.
2. Open your API keys/dashboard area.
3. Create or copy an API key.
4. Add it to `.env`.

Note: `WEATHER_API_KEY` is used by the code but is not currently listed in the main templates.

## Local Calendar

No API key needed.

```env
CALENDAR_ICS_PATH=data/calendar.ics
```

Use this if you want a local `.ics` calendar file.

## Legacy Or Currently Optional Names

These names may appear in an existing local `.env`, older notes, or old experiments. They are not the primary active integration variables in the current clients:

```env
GMAIL_CREDENTIALS=
GMAIL_TOKEN=
GOOGLE_CALENDAR_CREDENTIALS=
GOOGLE_CALENDAR_TOKEN=
NOTION_DATABASE_ID=
SPOTIFY_REDIRECT_URI=
OLLAMA_ENDPOINT=
DEBUG=
```

Keep them only if another local script you use still expects them. For the active Gmail and Calendar clients, prefer:

```env
GOOGLE_CLIENT_ID=
GOOGLE_CLIENT_SECRET=
GOOGLE_REFRESH_TOKEN=
```

For the active Ollama client, prefer:

```env
OLLAMA_BASE_URL=http://localhost:11434
```

## Suggested Full .env Template

Use this as a checklist. Fill only the integrations you need.

```env
# Core runtime
JARVIS_ENV=development
JARVIS_LOG_LEVEL=INFO
JARVIS_DASHBOARD_TOKEN=
JARVIS_SECRET_KEY=
JARVIS_ADMIN_USER=
JARVIS_ADMIN_PASSWORD=

# Local model runtime
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=deepseek-r1:8b

# Cloud fallback
CLOUD_LLM_FALLBACK_ENABLED=false
GEMINI_API_KEY=
GROQ_API_KEY=
OPENAI_API_KEY=
ANTHROPIC_API_KEY=

# Telegram
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=

# Google Calendar and Gmail API
GOOGLE_CLIENT_ID=
GOOGLE_CLIENT_SECRET=
GOOGLE_REFRESH_TOKEN=

# Notion
NOTION_API_KEY=

# Spotify
SPOTIFY_CLIENT_ID=
SPOTIFY_CLIENT_SECRET=
SPOTIFY_REFRESH_TOKEN=
SPOTIFY_REDIRECT_URI=http://127.0.0.1:8888/callback

# Home Assistant
HOME_ASSISTANT_URL=http://homeassistant.local:8123
HOME_ASSISTANT_TOKEN=

# GitHub
GITHUB_TOKEN=
GITHUB_DEFAULT_REPO=

# Generic IMAP/SMTP email
EMAIL_ADDRESS=
EMAIL_PASSWORD=
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
IMAP_HOST=imap.gmail.com

# Twilio WhatsApp
TWILIO_ACCOUNT_SID=
TWILIO_AUTH_TOKEN=
TWILIO_WHATSAPP_FROM=whatsapp:+14155238886

# Voice wake word
PORCUPINE_ACCESS_KEY=

# Web search
WEB_SEARCH_ENABLED=true
WEB_SEARCH_PROVIDER=auto
WEB_SEARCH_DEFAULT_MAX_RESULTS=5
WEB_SEARCH_SUMMARIZE_RESULTS=true
TAVILY_API_KEY=

# Weather
WEATHER_API_KEY=

# Local calendar
CALENDAR_ICS_PATH=data/calendar.ics
```

## Safe Verification

This prints whether values are set without printing the secret values:

```powershell
$keys = @(
  "JARVIS_DASHBOARD_TOKEN",
  "GEMINI_API_KEY",
  "GROQ_API_KEY",
  "OPENAI_API_KEY",
  "ANTHROPIC_API_KEY",
  "TELEGRAM_BOT_TOKEN",
  "TELEGRAM_CHAT_ID",
  "GOOGLE_CLIENT_ID",
  "GOOGLE_CLIENT_SECRET",
  "GOOGLE_REFRESH_TOKEN",
  "NOTION_API_KEY",
  "SPOTIFY_CLIENT_ID",
  "SPOTIFY_CLIENT_SECRET",
  "SPOTIFY_REFRESH_TOKEN",
  "HOME_ASSISTANT_TOKEN",
  "GITHUB_TOKEN",
  "EMAIL_PASSWORD",
  "TWILIO_ACCOUNT_SID",
  "TWILIO_AUTH_TOKEN",
  "PORCUPINE_ACCESS_KEY",
  "TAVILY_API_KEY",
  "WEATHER_API_KEY"
)

$envMap = @{}
Get-Content .env | ForEach-Object {
  if ($_ -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$') {
    $envMap[$matches[1]] = $matches[2].Trim().Trim('"').Trim("'")
  }
}

foreach ($key in $keys) {
  $value = $envMap[$key]
  if ([string]::IsNullOrWhiteSpace($value) -or $value -like "your_*" -or $value -like "*_here") {
    "{0}: missing or placeholder" -f $key
  } else {
    "{0}: set" -f $key
  }
}
```

Then run:

```powershell
.\run-jarvis.ps1 --health-check
```

If an integration does not load, check these three things:

1. The required `.env` variables are filled in.
2. The optional Python package for that integration is installed.
3. The provider key has the right scopes, permissions, billing status, and account/project.

## Rotation Checklist

Rotate a key when:

- It was committed to Git.
- It was pasted into a chat, issue, log, or screenshot.
- A provider dashboard shows unexpected usage.
- A teammate or tool no longer needs access.

Rotation steps:

1. Create a replacement key/token in the provider dashboard.
2. Update `.env`.
3. Restart Jarvis.
4. Confirm the integration works.
5. Revoke the old key.
6. Check usage and billing for suspicious activity.
