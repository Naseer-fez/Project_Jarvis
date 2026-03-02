"""Spotify integration via Spotify Web API (async aiohttp, OAuth PKCE refresh).

Required env vars:
    SPOTIFY_CLIENT_ID
    SPOTIFY_CLIENT_SECRET
    SPOTIFY_REFRESH_TOKEN  — From OAuth Authorization Code flow

Rules:
- Token refreshed on every call (no in-memory caching to avoid stale state)
- Fail gracefully if no active playback device
- Irreversible actions (play, create_playlist) gated as confirm-risk
"""

from __future__ import annotations

import os
from typing import Any

from integrations.base import BaseIntegration

_TOKEN_URL = "https://accounts.spotify.com/api/token"
_SPOTIFY_BASE = "https://api.spotify.com/v1"


class SpotifyIntegration(BaseIntegration):
    """Spotify Web API integration — playback control, search, playlists."""

    name = "spotify"
    description = "Control Spotify playback, search music, and manage playlists"
    required_config: list[str] = [
        "SPOTIFY_CLIENT_ID",
        "SPOTIFY_CLIENT_SECRET",
        "SPOTIFY_REFRESH_TOKEN",
    ]

    def is_available(self) -> bool:
        try:
            import aiohttp  # noqa: F401
        except Exception:
            self.unavailable_reason = "aiohttp not installed"
            return False
        if not all(bool(os.environ.get(k)) for k in self.required_config):
            missing = [k for k in self.required_config if not os.environ.get(k)]
            self.unavailable_reason = f"Missing env vars: {missing}"
            return False
        return True

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

    async def execute(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        args = args or {}
        try:
            token = await self._refresh_access_token()
            if tool_name == "play_track":
                return await self._play_track(token, args)
            if tool_name == "pause":
                return await self._pause(token, str(args.get("device_id", "") or ""))
            if tool_name == "search_track":
                return await self._search_track(
                    token,
                    query=str(args.get("query", "")),
                    limit=min(50, int(args.get("limit", 5) or 5)),
                )
            if tool_name == "get_current_track":
                return await self._get_current_track(token)
            if tool_name == "create_playlist":
                return await self._create_playlist(token, args)
            return {"success": False, "data": None, "error": f"Unknown tool: {tool_name}"}
        except Exception as exc:  # noqa: BLE001
            return {"success": False, "data": None, "error": str(exc)}

    # ── OAuth ─────────────────────────────────────────────────────────────────

    async def _refresh_access_token(self) -> str:
        import aiohttp
        import base64

        creds = base64.b64encode(
            f"{os.environ['SPOTIFY_CLIENT_ID']}:{os.environ['SPOTIFY_CLIENT_SECRET']}".encode()
        ).decode()
        headers = {"Authorization": f"Basic {creds}", "Content-Type": "application/x-www-form-urlencoded"}
        payload = {
            "grant_type": "refresh_token",
            "refresh_token": os.environ["SPOTIFY_REFRESH_TOKEN"],
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                _TOKEN_URL, data=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                data = await resp.json()
                if "access_token" not in data:
                    raise RuntimeError(f"Token refresh failed: {data.get('error', 'unknown')}")
                return data["access_token"]

    # ── Tool implementations ──────────────────────────────────────────────────

    async def _search_track(self, token: str, query: str, limit: int = 5) -> dict[str, Any]:
        import aiohttp

        if not query.strip():
            return {"success": False, "data": None, "error": "query is required"}

        headers = {"Authorization": f"Bearer {token}"}
        params = {"q": query, "type": "track", "limit": limit}

        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{_SPOTIFY_BASE}/search", headers=headers, params=params, timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                data = await resp.json()
                if resp.status != 200:
                    return {
                        "success": False,
                        "data": None,
                        "error": data.get("error", {}).get("message", str(resp.status)),
                    }

        tracks = [
            {
                "uri": t["uri"],
                "name": t["name"],
                "artist": ", ".join(a["name"] for a in t.get("artists", [])),
                "album": t.get("album", {}).get("name", ""),
                "duration_ms": t.get("duration_ms", 0),
            }
            for t in data.get("tracks", {}).get("items", [])
        ]
        return {"success": True, "data": {"tracks": tracks}, "error": None}

    async def _play_track(self, token: str, args: dict[str, Any]) -> dict[str, Any]:
        import aiohttp

        track_uri = str(args.get("track_uri", "") or "").strip()
        query = str(args.get("query", "") or "").strip()
        device_id = str(args.get("device_id", "") or "").strip()

        # If no URI provided, search first
        if not track_uri and query:
            search_result = await self._search_track(token, query, limit=1)
            if not search_result["success"] or not search_result["data"]["tracks"]:
                return {"success": False, "data": None, "error": f"No tracks found for: {query!r}"}
            track_uri = search_result["data"]["tracks"][0]["uri"]

        if not track_uri:
            return {"success": False, "data": None, "error": "Either track_uri or query is required"}

        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        payload = {"uris": [track_uri]}
        params = {}
        if device_id:
            params["device_id"] = device_id

        async with aiohttp.ClientSession() as session:
            async with session.put(
                f"{_SPOTIFY_BASE}/me/player/play",
                json=payload,
                headers=headers,
                params=params,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status == 204:
                    return {"success": True, "data": {"playing": track_uri}, "error": None}
                if resp.status == 404:
                    return {
                        "success": False,
                        "data": None,
                        "error": "No active Spotify device found. Open Spotify on a device first.",
                    }
                body = await resp.json()
                return {
                    "success": False,
                    "data": None,
                    "error": body.get("error", {}).get("message", str(resp.status)),
                }

    async def _pause(self, token: str, device_id: str = "") -> dict[str, Any]:
        import aiohttp

        headers = {"Authorization": f"Bearer {token}"}
        params = {}
        if device_id:
            params["device_id"] = device_id

        async with aiohttp.ClientSession() as session:
            async with session.put(
                f"{_SPOTIFY_BASE}/me/player/pause",
                headers=headers,
                params=params,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status == 204:
                    return {"success": True, "data": {"paused": True}, "error": None}
                if resp.status == 404:
                    return {"success": False, "data": None, "error": "No active device to pause."}
                body = await resp.json()
                return {
                    "success": False,
                    "data": None,
                    "error": body.get("error", {}).get("message", str(resp.status)),
                }

    async def _get_current_track(self, token: str) -> dict[str, Any]:
        import aiohttp

        headers = {"Authorization": f"Bearer {token}"}

        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{_SPOTIFY_BASE}/me/player/currently-playing",
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status == 204:
                    return {"success": True, "data": {"playing": False, "track": None}, "error": None}
                if resp.status != 200:
                    return {"success": False, "data": None, "error": f"HTTP {resp.status}"}
                data = await resp.json()

        item = data.get("item") or {}
        return {
            "success": True,
            "data": {
                "playing": data.get("is_playing", False),
                "track": {
                    "uri": item.get("uri"),
                    "name": item.get("name"),
                    "artist": ", ".join(a["name"] for a in item.get("artists", [])),
                    "album": item.get("album", {}).get("name"),
                    "progress_ms": data.get("progress_ms"),
                    "duration_ms": item.get("duration_ms"),
                }
                if item
                else None,
            },
            "error": None,
        }

    async def _create_playlist(self, token: str, args: dict[str, Any]) -> dict[str, Any]:
        import aiohttp

        name = str(args.get("name", "")).strip()
        if not name:
            return {"success": False, "data": None, "error": "name is required"}

        description = str(args.get("description", "") or "")
        public = bool(args.get("public", False))

        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

        # Get current user ID first
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{_SPOTIFY_BASE}/me", headers=headers, timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status != 200:
                    return {"success": False, "data": None, "error": "Could not retrieve Spotify user ID"}
                user_data = await resp.json()
                user_id = user_data["id"]

            payload = {"name": name[:100], "description": description[:300], "public": public}
            async with session.post(
                f"{_SPOTIFY_BASE}/users/{user_id}/playlists",
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                data = await resp.json()
                if resp.status not in (200, 201):
                    return {
                        "success": False,
                        "data": None,
                        "error": data.get("error", {}).get("message", str(resp.status)),
                    }
                return {
                    "success": True,
                    "data": {
                        "playlist_id": data["id"],
                        "name": data["name"],
                        "url": data.get("external_urls", {}).get("spotify"),
                    },
                    "error": None,
                }


__all__ = ["SpotifyIntegration"]
