# File Report: spotify.py
**Path**: `d:\AI\Jarvis\integrations\clients\spotify.py`
**Role**: Data Model Analyst

## Analysis Summary
This file has been analyzed for schemas, DTOs, state objects, config variables, and dependencies.

## Dependencies
- __future__.annotations
- os
- typing.Any
- integrations.base.BaseIntegration
- aiohttp
- base64
- aiohttp
- aiohttp
- aiohttp
- aiohttp
- aiohttp
- aiohttp

## Classes and State Objects
### `SpotifyIntegration`
**Variables**: name, description
**Methods**: is_available, get_tools, execute, _refresh_access_token, _search_track, _play_track, _pause, _get_current_track, _create_playlist

## Tool Schemas / DTOs
```python
    def get_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "play_track",
                "description": "Play a Spotify track by URI or search query on the active device",
                "risk": "confirm",
                "args": {
                    "track_uri": {
                        "type": "string",
                        "description": "Spotify track URI (spotify:track:...) — preferred",
                        "default": "",
                    },
                    "query": {
                        "type": "string",
                        "description": "If track_uri is empty, search by this query and play first result",
                        "default": "",
                    },
                    "device_id": {
                        "type": "string",
                        "description": "Optional Spotify device ID; uses active device if omitted",
                        "default": "",
                    },
                },
                "required_args": [],
            },
            {
                "name": "pause",
                "description": "Pause the current Spotify playback",
                "risk": "low",
                "args": {
                    "device_id": {"type": "string", "default": ""},
                },
                "required_args": [],
            },
            {
                "name": "search_track",
                "description": "Search Spotify for tracks matching a query",
                "risk": "low",
                "args": {
                    "query": {"type": "string", "description": "Search query (artist, track, album)"},
                    "limit": {"type": "integer", "default": 5},
                },
                "required_args": ["query"],
            },
            {
                "name": "get_current_track",
                "description": "Get the currently playing track on Spotify",
                "risk": "low",
                "args": {},
                "required_args": [],
            },
            {
                "name": "create_playlist",
                "description": "Create a new Spotify playlist for the current user",
                "risk": "confirm",
                "args": {
                    "name": {"type": "string", "description": "Playlist name"},
                    "description": {"type": "string", "default": ""},
                    "public": {"type": "boolean", "default": False},
                },
                "required_args": ["name"],
            },
        ]

```

## Assumptions & API Contracts
1. Config vars are expected in environment variables.
2. Schema validation is typically deferred to the registry or client implementation.