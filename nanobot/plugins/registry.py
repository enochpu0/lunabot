"""Auto-discovery for built-in plugin modules and external plugin packages."""

from __future__ import annotations

import importlib
import pkgutil
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from nanobot.plugins.base import BasePlugin

_INTERNAL = frozenset({"base", "registry", "manager"})


def discover_plugin_names() -> list[str]:
    """Return all built-in plugin module names by scanning the package (zero imports)."""
    import nanobot.plugins as pkg

    return [
        name
        for _, name, ispkg in pkgutil.iter_modules(pkg.__path__)
        if name not in _INTERNAL and not ispkg
    ]


def load_plugin_class(module_name: str) -> type[BasePlugin]:
    """Import *module_name* and return the first BasePlugin subclass found."""
    from nanobot.plugins.base import BasePlugin as _Base

    mod = importlib.import_module(f"nanobot.plugins.{module_name}")
    for attr in dir(mod):
        obj = getattr(mod, attr)
        if isinstance(obj, type) and issubclass(obj, _Base) and obj is not _Base:
            return obj
    raise ImportError(f"No BasePlugin subclass in nanobot.plugins.{module_name}")


def discover_external_plugins() -> dict[str, type[BasePlugin]]:
    """Discover external plugins registered via entry_points."""
    from importlib.metadata import entry_points

    plugins: dict[str, type[BasePlugin]] = {}
    for ep in entry_points(group="nanobot.plugins"):
        try:
            cls = ep.load()
            plugins[ep.name] = cls
        except Exception as e:
            logger.warning("Failed to load plugin '{}': {}", ep.name, e)
    return plugins


def discover_all_plugins() -> dict[str, type[BasePlugin]]:
    """Return all plugins: built-in (pkgutil) merged with external (entry_points).

    Built-in plugins take priority — an external plugin cannot shadow a built-in name.
    """
    builtin: dict[str, type[BasePlugin]] = {}
    for modname in discover_plugin_names():
        try:
            builtin[modname] = load_plugin_class(modname)
        except ImportError as e:
            logger.debug("Skipping built-in plugin '{}': {}", modname, e)

    external = discover_external_plugins()
    shadowed = set(external) & set(builtin)
    if shadowed:
        logger.warning("External plugin(s) shadowed by built-in plugins (ignored): {}", shadowed)

    return {**external, **builtin}