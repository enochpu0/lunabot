"""Plugin manager for coordinating nanobot plugins."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from loguru import logger

from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.plugins.base import BasePlugin
from nanobot.plugins.registry import discover_all_plugins

if TYPE_CHECKING:
    from nanobot.agent.hook import AgentHook
    from nanobot.config.schema import PluginsConfig


class PluginManager:
    """
    Manages plugins and coordinates lifecycle and message hooks.

    Responsibilities:
    - Initialize discovered plugins
    - Start/stop plugins
    - Route message bus hooks
    """

    def __init__(self, config: "PluginsConfig | dict[str, Any] | None" = None, bus: MessageBus | None = None):
        self.config = config
        self.bus = bus or MessageBus()
        self.plugins: dict[str, BasePlugin] = {}

        self._init_plugins()

    def _init_plugins(self) -> None:
        """Initialize plugins discovered via pkgutil scan + entry_points plugins."""
        from nanobot.config.schema import PluginsConfig

        for name, cls in discover_all_plugins().items():
            section = None
            if isinstance(self.config, PluginsConfig):
                section = getattr(self.config, name, None)
            elif isinstance(self.config, dict):
                section = self.config.get(name)
            if section is None:
                continue
            enabled = (
                section.get("enabled", False)
                if isinstance(section, dict)
                else getattr(section, "enabled", False)
            )
            if not enabled:
                continue
            try:
                plugin = cls(section, self.bus)
                plugin.is_enabled = True
                self.plugins[name] = plugin
                logger.info("{} plugin enabled", cls.display_name)
            except Exception as e:
                logger.warning("{} plugin not available: {}", name, e)

    async def _start_plugin(self, name: str, plugin: BasePlugin) -> None:
        """Start a plugin and log any exceptions."""
        try:
            await plugin.start()
        except Exception as e:
            logger.error("Failed to start plugin {}: {}", name, e)

    async def start_all(self) -> None:
        """Start all plugins."""
        if not self.plugins:
            logger.debug("No plugins enabled")
            return

        # Start plugins
        tasks = []
        for name, plugin in self.plugins.items():
            logger.info("Starting {} plugin...", name)
            tasks.append(asyncio.create_task(self._start_plugin(name, plugin)))

        await asyncio.gather(*tasks, return_exceptions=True)

    async def dispatch_inbound(self, msg: InboundMessage) -> InboundMessage:
        """Run inbound message through all plugin hooks, returning the (possibly modified) message.

        AgentLoop calls this instead of consuming directly from the bus.
        """
        result = msg
        for plugin in self.plugins.values():
            if plugin.is_enabled:
                new_msg = await self._safe_hook(plugin.on_inbound, result)
                if new_msg is not None:
                    result = new_msg
        return result

    async def dispatch_outbound(self, msg: OutboundMessage) -> None:
        """Run outbound message through all plugin hooks (fire-and-forget)."""
        for plugin in self.plugins.values():
            if plugin.is_enabled:
                await self._safe_hook(plugin.on_outbound, msg)

    async def stop_all(self) -> None:
        """Stop all plugins."""
        logger.info("Stopping all plugins...")

        for name, plugin in self.plugins.items():
            try:
                await plugin.stop()
                logger.info("Stopped {} plugin", name)
            except Exception as e:
                logger.error("Error stopping {}: {}", name, e)

    async def _safe_hook(self, hook, msg: Any) -> Any:
        """Call a hook safely, logging any exceptions. Returns hook result or None on error."""
        try:
            return await hook(msg)
        except Exception as e:
            logger.warning("Plugin hook {.__name__} failed: {}", hook, e)
            return None

    def get_plugin(self, name: str) -> BasePlugin | None:
        """Get a plugin by name."""
        return self.plugins.get(name)

    def get_status(self) -> dict[str, Any]:
        """Get status of all plugins."""
        return {
            name: {
                "enabled": plugin.is_enabled,
                "running": plugin.is_enabled,
            }
            for name, plugin in self.plugins.items()
        }

    @property
    def enabled_plugins(self) -> list[str]:
        """Get list of enabled plugin names."""
        return list(self.plugins.keys())

    @property
    def enabled_agent_hooks(self) -> list["AgentHook"]:
        """Get enabled plugins that also implement AgentHook (for runner integration)."""
        from nanobot.agent.hook import AgentHook
        return [p for p in self.plugins.values() if isinstance(p, AgentHook)]