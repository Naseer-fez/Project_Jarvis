"""Home Assistant integration via the REST API.

Required env vars:
    HOME_ASSISTANT_URL
    HOME_ASSISTANT_TOKEN
"""

from __future__ import annotations

import os
import time
from typing import Any
from urllib.parse import quote

from integrations.base import BaseIntegration

_ENTITY_CACHE_TTL_SECONDS = 60
_SENSITIVE_DOMAINS = {"lock", "alarm_control_panel"}


class HomeAssistantIntegration(BaseIntegration):
    """Read entity state and call Home Assistant services."""

    name = "home_assistant"
    description = "Inspect entities and control smart-home devices through Home Assistant"
    required_config: list[str] = ["HOME_ASSISTANT_URL", "HOME_ASSISTANT_TOKEN"]

    def __init__(self, config: Any | None = None) -> None:
        super().__init__(config=config)
        self._entity_cache: list[dict[str, Any]] = []
        self._entity_cache_at: float = 0.0

    def is_available(self) -> bool:
        try:
            import aiohttp  # noqa: F401
        except Exception:
            self.unavailable_reason = "aiohttp not installed"
            return False

        if not all(bool(os.environ.get(key)) for key in self.required_config):
            missing = [key for key in self.required_config if not os.environ.get(key)]
            self.unavailable_reason = f"Missing env vars: {missing}"
            return False
        return True

    def get_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "get_entity_state",
                "description": "Get the current state and friendly name for a Home Assistant entity",
                "risk": "low",
                "args": {
                    "entity_id": {"type": "string", "description": "Entity ID like light.kitchen"},
                },
                "required_args": ["entity_id"],
            },
            {
                "name": "turn_on_entity",
                "description": "Turn on a Home Assistant light, switch, fan, or similar entity",
                "risk": "confirm",
                "args": {
                    "entity_id": {"type": "string", "description": "Single entity ID", "default": ""},
                    "entity_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of entity IDs",
                        "default": [],
                    },
                    "area_id": {"type": "string", "description": "Optional Home Assistant area ID", "default": ""},
                    "device_id": {"type": "string", "description": "Optional Home Assistant device ID", "default": ""},
                    "domain": {"type": "string", "description": "Required when targeting an area or device", "default": ""},
                    "service_data": {
                        "type": "object",
                        "description": "Optional extra service data like brightness_pct",
                        "default": {},
                    },
                },
                "required_args": [],
            },
            {
                "name": "turn_off_entity",
                "description": "Turn off a Home Assistant light, switch, fan, or similar entity",
                "risk": "confirm",
                "args": {
                    "entity_id": {"type": "string", "description": "Single entity ID", "default": ""},
                    "entity_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of entity IDs",
                        "default": [],
                    },
                    "area_id": {"type": "string", "description": "Optional Home Assistant area ID", "default": ""},
                    "device_id": {"type": "string", "description": "Optional Home Assistant device ID", "default": ""},
                    "domain": {"type": "string", "description": "Required when targeting an area or device", "default": ""},
                    "service_data": {
                        "type": "object",
                        "description": "Optional extra service data",
                        "default": {},
                    },
                },
                "required_args": [],
            },
            {
                "name": "toggle_entity",
                "description": "Toggle a Home Assistant light, switch, fan, or similar entity",
                "risk": "confirm",
                "args": {
                    "entity_id": {"type": "string", "description": "Single entity ID", "default": ""},
                    "entity_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of entity IDs",
                        "default": [],
                    },
                    "area_id": {"type": "string", "description": "Optional Home Assistant area ID", "default": ""},
                    "device_id": {"type": "string", "description": "Optional Home Assistant device ID", "default": ""},
                    "domain": {"type": "string", "description": "Required when targeting an area or device", "default": ""},
                    "service_data": {
                        "type": "object",
                        "description": "Optional extra service data",
                        "default": {},
                    },
                },
                "required_args": [],
            },
            {
                "name": "set_thermostat",
                "description": "Set a climate entity target temperature",
                "risk": "confirm",
                "args": {
                    "entity_id": {"type": "string", "description": "Climate entity ID like climate.living_room"},
                    "temperature": {"type": "number", "description": "Target temperature"},
                    "hvac_mode": {
                        "type": "string",
                        "description": "Optional HVAC mode like heat, cool, or auto",
                        "default": "",
                    },
                },
                "required_args": ["entity_id", "temperature"],
            },
            {
                "name": "call_service",
                "description": "Call a specific Home Assistant service for a targeted entity, area, or device",
                "risk": "confirm",
                "args": {
                    "domain": {"type": "string", "description": "Service domain like light or media_player"},
                    "service": {"type": "string", "description": "Service name like turn_on or volume_set"},
                    "entity_id": {"type": "string", "description": "Optional single entity ID", "default": ""},
                    "entity_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of entity IDs",
                        "default": [],
                    },
                    "area_id": {"type": "string", "description": "Optional area ID", "default": ""},
                    "device_id": {"type": "string", "description": "Optional device ID", "default": ""},
                    "service_data": {
                        "type": "object",
                        "description": "Optional extra Home Assistant service fields",
                        "default": {},
                    },
                },
                "required_args": ["domain", "service"],
            },
            {
                "name": "list_entities",
                "description": "List Home Assistant entities, optionally filtered by domain",
                "risk": "low",
                "args": {
                    "domain": {"type": "string", "description": "Optional domain filter like light", "default": ""},
                    "include_attributes": {
                        "type": "boolean",
                        "description": "Include raw Home Assistant attributes in the response",
                        "default": False,
                    },
                },
                "required_args": [],
            },
        ]

    async def execute(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        args = args or {}
        try:
            if tool_name == "get_entity_state":
                return await self._get_entity_state(args)
            if tool_name == "turn_on_entity":
                return await self._entity_service("turn_on", args)
            if tool_name == "turn_off_entity":
                return await self._entity_service("turn_off", args)
            if tool_name == "toggle_entity":
                return await self._entity_service("toggle", args)
            if tool_name == "set_thermostat":
                return await self._set_thermostat(args)
            if tool_name == "call_service":
                return await self._call_service(args)
            if tool_name == "list_entities":
                return await self._list_entities(args)
            return {"success": False, "data": None, "error": f"Unknown tool: {tool_name}"}
        except Exception as exc:  # noqa: BLE001
            return {"success": False, "data": None, "error": str(exc)}

    def _base_url(self) -> str:
        value = str(os.environ["HOME_ASSISTANT_URL"]).strip().rstrip("/")
        if value.endswith("/api"):
            value = value[:-4]
        return value

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {os.environ['HOME_ASSISTANT_TOKEN']}",
            "Content-Type": "application/json",
        }

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_payload: dict[str, Any] | None = None,
    ) -> Any:
        import aiohttp

        url = f"{self._base_url()}{path}"
        timeout = aiohttp.ClientTimeout(total=15)
        async with aiohttp.ClientSession() as session:
            request_fn = getattr(session, method.lower())
            async with request_fn(
                url,
                headers=self._headers(),
                json=json_payload,
                timeout=timeout,
            ) as resp:
                data = await self._read_response(resp)
                if resp.status >= 400:
                    raise RuntimeError(self._extract_error_message(resp.status, data))
                return data

    async def _read_response(self, resp: Any) -> Any:
        if getattr(resp, "status", None) == 204:
            return {}
        return await resp.json(content_type=None)

    def _extract_error_message(self, status: int, data: Any) -> str:
        if isinstance(data, dict):
            for key in ("message", "error", "detail"):
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    return value
        if isinstance(data, str) and data.strip():
            return data
        return f"Home Assistant API request failed with HTTP {status}"

    def _invalidate_entity_cache(self) -> None:
        self._entity_cache = []
        self._entity_cache_at = 0.0

    async def _get_states(self, *, force_refresh: bool = False) -> list[dict[str, Any]]:
        is_fresh = (time.monotonic() - self._entity_cache_at) < _ENTITY_CACHE_TTL_SECONDS
        if not force_refresh and self._entity_cache and is_fresh:
            return self._entity_cache

        data = await self._request("get", "/api/states")
        if not isinstance(data, list):
            raise RuntimeError("Unexpected Home Assistant states response")

        self._entity_cache = [item for item in data if isinstance(item, dict)]
        self._entity_cache_at = time.monotonic()
        return self._entity_cache

    def _extract_entity_ids(self, args: dict[str, Any]) -> list[str]:
        ids: list[str] = []

        raw_entity_id = args.get("entity_id", "")
        if isinstance(raw_entity_id, list):
            ids.extend(str(item).strip() for item in raw_entity_id if str(item).strip())
        else:
            value = str(raw_entity_id or "").strip()
            if value:
                ids.append(value)

        raw_entity_ids = args.get("entity_ids", [])
        if isinstance(raw_entity_ids, list):
            ids.extend(str(item).strip() for item in raw_entity_ids if str(item).strip())

        deduped: list[str] = []
        seen: set[str] = set()
        for entity_id in ids:
            if entity_id not in seen:
                deduped.append(entity_id)
                seen.add(entity_id)
        return deduped

    def _build_target(self, args: dict[str, Any]) -> dict[str, Any]:
        target: dict[str, Any] = {}
        entity_ids = self._extract_entity_ids(args)

        if entity_ids:
            target["entity_id"] = entity_ids[0] if len(entity_ids) == 1 else entity_ids

        area_id = str(args.get("area_id", "") or "").strip()
        if area_id:
            target["area_id"] = area_id

        device_id = str(args.get("device_id", "") or "").strip()
        if device_id:
            target["device_id"] = device_id

        return target

    def _infer_domain(self, args: dict[str, Any]) -> str:
        explicit = str(args.get("domain", "") or "").strip().lower()
        if explicit:
            return explicit

        entity_ids = self._extract_entity_ids(args)
        if not entity_ids:
            return ""
        entity_id = entity_ids[0]
        return entity_id.split(".", 1)[0].strip().lower() if "." in entity_id else ""

    def _contains_sensitive_domain(self, args: dict[str, Any], *, explicit_domain: str = "") -> bool:
        domains: set[str] = set()
        if explicit_domain:
            domains.add(explicit_domain.strip().lower())

        for entity_id in self._extract_entity_ids(args):
            if "." in entity_id:
                domains.add(entity_id.split(".", 1)[0].strip().lower())

        return any(domain in _SENSITIVE_DOMAINS for domain in domains)

    def _normalize_service_data(self, args: dict[str, Any]) -> dict[str, Any]:
        service_data = args.get("service_data", {})
        if isinstance(service_data, dict):
            return dict(service_data)
        return {}

    def _format_entity(self, item: dict[str, Any], *, include_attributes: bool = False) -> dict[str, Any]:
        entity_id = str(item.get("entity_id", ""))
        attributes = item.get("attributes") if isinstance(item.get("attributes"), dict) else {}
        payload = {
            "entity_id": entity_id,
            "domain": entity_id.split(".", 1)[0] if "." in entity_id else "",
            "state": item.get("state"),
            "friendly_name": attributes.get("friendly_name") or entity_id,
            "last_changed": item.get("last_changed"),
            "last_updated": item.get("last_updated"),
        }
        if include_attributes:
            payload["attributes"] = attributes
        return payload

    async def _get_entity_state(self, args: dict[str, Any]) -> dict[str, Any]:
        entity_id = str(args.get("entity_id", "") or "").strip()
        if not entity_id:
            return {"success": False, "data": None, "error": "entity_id is required"}

        states = await self._get_states()
        for item in states:
            if str(item.get("entity_id", "")) == entity_id:
                return {"success": True, "data": self._format_entity(item, include_attributes=True), "error": None}

        encoded_entity_id = quote(entity_id, safe="")
        data = await self._request("get", f"/api/states/{encoded_entity_id}")
        if not isinstance(data, dict):
            raise RuntimeError("Unexpected Home Assistant entity response")
        return {"success": True, "data": self._format_entity(data, include_attributes=True), "error": None}

    async def _entity_service(self, service: str, args: dict[str, Any]) -> dict[str, Any]:
        domain = self._infer_domain(args)
        target = self._build_target(args)
        if not target:
            return {
                "success": False,
                "data": None,
                "error": "Provide entity_id, entity_ids, area_id, or device_id",
            }
        if not domain:
            return {
                "success": False,
                "data": None,
                "error": "domain is required when the target is an area or device",
            }
        if self._contains_sensitive_domain(args, explicit_domain=domain):
            return {
                "success": False,
                "data": None,
                "error": "Sensitive domains must use explicit confirm-gated services instead of the convenience helpers",
            }

        payload = self._normalize_service_data(args)
        payload.update(target)
        data = await self._request("post", f"/api/services/{domain}/{service}", json_payload=payload)
        self._invalidate_entity_cache()
        return {
            "success": True,
            "data": {
                "service": f"{domain}.{service}",
                "result": self._format_service_result(data),
            },
            "error": None,
        }

    async def _set_thermostat(self, args: dict[str, Any]) -> dict[str, Any]:
        entity_id = str(args.get("entity_id", "") or "").strip()
        if not entity_id:
            return {"success": False, "data": None, "error": "entity_id is required"}
        if "temperature" not in args:
            return {"success": False, "data": None, "error": "temperature is required"}

        payload: dict[str, Any] = {
            "entity_id": entity_id,
            "temperature": args["temperature"],
        }
        hvac_mode = str(args.get("hvac_mode", "") or "").strip()
        if hvac_mode:
            payload["hvac_mode"] = hvac_mode

        data = await self._request("post", "/api/services/climate/set_temperature", json_payload=payload)
        self._invalidate_entity_cache()
        return {
            "success": True,
            "data": {
                "service": "climate.set_temperature",
                "result": self._format_service_result(data),
            },
            "error": None,
        }

    async def _call_service(self, args: dict[str, Any]) -> dict[str, Any]:
        domain = str(args.get("domain", "") or "").strip().lower()
        service = str(args.get("service", "") or "").strip()
        if not domain:
            return {"success": False, "data": None, "error": "domain is required"}
        if not service:
            return {"success": False, "data": None, "error": "service is required"}

        target = self._build_target(args)
        if not target:
            return {
                "success": False,
                "data": None,
                "error": "Targeted service calls require entity_id, entity_ids, area_id, or device_id",
            }

        payload = self._normalize_service_data(args)
        payload.update(target)
        data = await self._request("post", f"/api/services/{domain}/{service}", json_payload=payload)
        self._invalidate_entity_cache()
        return {
            "success": True,
            "data": {
                "service": f"{domain}.{service}",
                "result": self._format_service_result(data),
            },
            "error": None,
        }

    async def _list_entities(self, args: dict[str, Any]) -> dict[str, Any]:
        domain = str(args.get("domain", "") or "").strip().lower()
        include_attributes = bool(args.get("include_attributes", False))
        states = await self._get_states()

        if domain:
            states = [item for item in states if str(item.get("entity_id", "")).startswith(f"{domain}.")]

        entities = [self._format_entity(item, include_attributes=include_attributes) for item in states]
        return {
            "success": True,
            "data": {
                "entities": entities,
                "count": len(entities),
                "cached": True,
            },
            "error": None,
        }

    def _format_service_result(self, data: Any) -> Any:
        if not isinstance(data, list):
            return data

        formatted: list[dict[str, Any]] = []
        for item in data:
            if isinstance(item, dict) and item.get("entity_id"):
                formatted.append(self._format_entity(item, include_attributes=False))
            elif isinstance(item, dict):
                formatted.append(dict(item))
        return formatted


__all__ = ["HomeAssistantIntegration"]
