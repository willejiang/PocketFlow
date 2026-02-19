"""
Cookbook Adapter System

Provides a standard interface for integrating any cookbook with the unified agent.
"""

from .base import (
    CookbookAdapter,
    AdapterAction,
    AdapterRegistry,
    get_adapter_registry,
)
from .discovery import (
    discover_cookbooks,
    load_cookbook_adapter,
    get_cookbook_info,
    CookbookInfo,
)
from .system_tools import (
    SystemToolsAdapter,
    get_system_tools_adapter,
)

__all__ = [
    "CookbookAdapter",
    "AdapterAction",
    "AdapterRegistry",
    "get_adapter_registry",
    "discover_cookbooks",
    "load_cookbook_adapter",
    "get_cookbook_info",
    "CookbookInfo",
    "SystemToolsAdapter",
    "get_system_tools_adapter",
]
