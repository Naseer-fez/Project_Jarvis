"""Manifest-driven plugin catalog for Jarvis extensions."""

from .manifest import (
    PluginCatalog,
    PluginManifest,
    PluginManifestError,
    load_plugin_manifest,
)

__all__ = [
    "PluginCatalog",
    "PluginManifest",
    "PluginManifestError",
    "load_plugin_manifest",
]
