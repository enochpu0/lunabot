"""Nanobot plugins module with plugin architecture."""

from nanobot.plugins.base import BasePlugin
from nanobot.plugins.manager import PluginManager
from nanobot.plugins.registry import discover_all_plugins, discover_plugin_names, load_plugin_class

__all__ = [
    "BasePlugin",
    "PluginManager",
    "discover_all_plugins",
    "discover_plugin_names",
    "load_plugin_class",
]