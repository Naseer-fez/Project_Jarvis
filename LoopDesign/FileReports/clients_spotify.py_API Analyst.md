# `spotify.py` - API Analyst Report

## Overview
Spotify integration via the Spotify Web API, using OAuth PKCE refresh. 

## Endpoints / Tools
1. `play_track`
   - Description: Play a Spotify track by URI or search query on the active device.
   - Risk: confirm (write)
   - Arguments: `track_uri` (string), `query` (string), `device_id` (string).
2. `pause`
   - Description: Pause the current Spotify playback.
   - Risk: low (read-only)
   - Arguments: `device_id` (string).
3. `search_track`
   - Description: Search Spotify for tracks matching a query.
   - Risk: low (read-only)
   - Arguments: `query` (string, required), `limit` (integer, default 5).
4. `get_current_track`
   - Description: Get the currently playing track on Spotify.
   - Risk: low (read-only)
   - Arguments: None.
5. `create_playlist`
   - Description: Create a new Spotify playlist for the current user.
   - Risk: confirm (write)
   - Arguments: `name` (string, required), `description` (string), `public` (boolean, default False).

## External Contracts / Dependencies
- Requires `SPOTIFY_CLIENT_ID`, `SPOTIFY_CLIENT_SECRET`, `SPOTIFY_REFRESH_TOKEN`.
- Token URL: `https://accounts.spotify.com/api/token`.
- Base API URL: `https://api.spotify.com/v1`.
- Needs `aiohttp` for async HTTP execution.

## Assumptions
- Explicitly states: "Fail gracefully if no active playback device" (HTTP 404).
- Does not cache tokens in memory; refreshes the token on every single call to avoid stale state.
- If `play_track` is called with a `query` and no `track_uri`, it performs a `search_track` internally with `limit=1` and plays the first result.
- Implicitly depends on user having an active Spotify device for `play_track` unless `device_id` is supplied (though it usually also requires the device to be active).
- `create_playlist` requires an initial `/me` request to resolve the user ID before posting to `/users/{user_id}/playlists`. Placed limits on name (100) and description (300).
