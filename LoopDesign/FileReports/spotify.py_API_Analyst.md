# clients/spotify.py API Analyst Report

## Overview
Spotify Web API integration providing async playback control and playlist management.

## API Contracts & Methods
- `SpotifyIntegration(BaseIntegration)`
  - OAuth token is refreshed dynamically on *every* call (no caching) to avoid stale states.

## Tools Exposed
- `play_track(track_uri, query, device_id)` [Risk: `confirm`]
  - Automatically delegates to search if `query` is provided instead of `track_uri`.
  - Gracefully handles HTTP 404 (No active device).
- `pause(device_id)` [Risk: `low`]
- `search_track(query, limit=5)` [Risk: `low`]
- `get_current_track()` [Risk: `low`]
- `create_playlist(name, description, public=False)` [Risk: `confirm`]
  - Looks up the user ID dynamically before creating the playlist.

## Configuration Variables
- `SPOTIFY_CLIENT_ID`
- `SPOTIFY_CLIENT_SECRET`
- `SPOTIFY_REFRESH_TOKEN`

## Assumptions & Constants
- Uses PKCE/Auth Code grant dynamically.
- Needs an active device to play/pause.

## Dependencies
- `aiohttp`, `base64`

## Prompts
- None.
