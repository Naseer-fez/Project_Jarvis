# Documentation Report: clients/spotify.py

## Assumptions
- Uses Spotify Web API natively.
- Handles authorization via `refresh_token` and refreshes on *every call*.
- Fails gracefully if there's no "active device" (returns 404).
- Creating playlist requires fetching the current user ID first.
- If playing by query, performs search first and picks the very first result.

## Schema / API Contract
- Tools: `play_track`, `pause`, `search_track`, `get_current_track`, `create_playlist`.

## Dependencies
- `aiohttp` (external)
- `os`, `base64` (stdlib)

## Configuration Variables
- `SPOTIFY_CLIENT_ID`
- `SPOTIFY_CLIENT_SECRET`
- `SPOTIFY_REFRESH_TOKEN`

## Prompts
None.
