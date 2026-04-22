"""Base plugin interface for nanobot extensions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from loguru import logger

from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus


class BasePlugin(ABC):
    """
    Abstract base class for nanobot plugin implementations.

    Plugins extend nanobot's functionality through hooks into the agent
    loop, message bus, and lifecycle events.
    """

    name: str = "base"
    display_name: str = "Base Plugin"

    def __init__(self, config: Any, bus: MessageBus):
        """
        Initialize the plugin.

        Args:
            config: Plugin-specific configuration.
            bus: The message bus for communication.
        """
        self.config = config
        self.bus = bus
        self._enabled = False

    @abstractmethod
    async def start(self) -> None:
        """Start the plugin and perform any setup."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the plugin and clean up resources."""
        pass

    async def on_inbound(self, msg: InboundMessage) -> None:
        """
        Hook called when an inbound message is published to the bus.

        Override in subclasses to inspect or modify messages.
        By default, messages pass through unchanged.
        """
        pass

    async def on_outbound(self, msg: OutboundMessage) -> None:
        """
        Hook called when an outbound message is published to the bus.

        Override in subclasses to inspect or modify outgoing messages.
        By default, messages pass through unchanged.
        """
        pass

    async def on_agent_start(self) -> None:
        """Hook called before the agent loop starts."""
        pass

    async def on_agent_stop(self) -> None:
        """Hook called after the agent loop stops."""
        pass

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        """Return default config for onboard. Override in plugins to auto-populate config."""
        return {"enabled": False}

    @property
    def is_enabled(self) -> bool:
        """Check if the plugin is enabled."""
        return self._enabled

    @is_enabled.setter
    def is_enabled(self, value: bool) -> None:
        """Set plugin enabled state."""
        self._enabled = value