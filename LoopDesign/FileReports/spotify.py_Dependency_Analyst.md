# File Report: spotify.py
## Role: Dependency Analyst

### 1. Library Requirements
- `aiohttp` (Third-party)
- `base64` (implicit in refresh logic), `os`, `typing` (Standard Library)
- `integrations.base` (Local)

### 2. Service Dependencies
- Spotify Web API (`https://api.spotify.com/v1`)
- Spotify Accounts API for token refresh (`https://accounts.spotify.com/api/token`)

### 3. Hidden Execution Links
- Requires the user to actively run a Spotify device for playback tools to work.
- If `track_uri` is omitted but `query` is provided in `play_track`, it invokes `search_track` internally to find the first result, chaining API calls.
- Creates playlists by querying `/me` first to retrieve the User ID, then POSTing to `/users/{user_id}/playlists`.

### 4. Assumptions & API Contracts
- OAuth Authorization Code flow is expected to have occurred ahead of time. It strictly uses `grant_type="refresh_token"`.
- Token is refreshed synchronously on every tool call.
- Tracks and playlists strings are truncated/capped before submission (`name[:100]`, `description[:300]`).

### 5. Configuration Variables
- `SPOTIFY_CLIENT_ID`
- `SPOTIFY_CLIENT_SECRET`
- `SPOTIFY_REFRESH_TOKEN`

### 6. Prompts Found
- None.
