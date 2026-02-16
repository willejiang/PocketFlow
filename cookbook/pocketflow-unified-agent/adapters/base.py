"""
Base classes for the cookbook adapter system.

CookbookAdapter: Interface that any cookbook can implement to expose capabilities
AdapterAction: Represents a single action that an adapter provides
AdapterRegistry: Manages all loaded adapters
"""

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path


@dataclass
class AdapterAction:
    """
    Represents a single action that a cookbook adapter provides.
    
    Attributes:
        name: Unique action name within this adapter
        description: Human-readable description
        parameters: Dict of parameter definitions {name: {type, description, required, default}}
        handler: Optional callable that executes the action
    """
    name: str
    description: str
    parameters: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    handler: Optional[Callable] = None
    
    def format_for_prompt(self) -> str:
        """Format action for LLM prompt."""
        lines = [f"[{self.name}]", f"  Description: {self.description}"]
        
        if self.parameters:
            lines.append("  Parameters:")
            for pname, pinfo in self.parameters.items():
                ptype = pinfo.get("type", "any")
                pdesc = pinfo.get("description", "")
                required = pinfo.get("required", True)
                default = pinfo.get("default")
                
                req_str = "required" if required else f"optional, default={default}"
                lines.append(f"    - {pname} ({ptype}, {req_str}): {pdesc}")
        
        return "\n".join(lines)
    
    def validate_params(self, params: Dict[str, Any]) -> tuple:
        """
        Validate parameters against schema.
        Returns (is_valid, error_message).
        """
        for pname, pinfo in self.parameters.items():
            required = pinfo.get("required", True)
            if required and pname not in params:
                return False, f"Missing required parameter: {pname}"
        return True, None


class CookbookAdapter(ABC):
    """
    Abstract base class for cookbook adapters.
    
    Each cookbook can provide an adapter that exposes its capabilities
    to the unified agent. The adapter defines what actions are available
    and how to execute them.
    
    To create an adapter for a cookbook:
    1. Create a class that inherits from CookbookAdapter
    2. Implement the required properties and methods
    3. Place it in the cookbook directory as `adapter.py` or register via manifest
    """
    
    def __init__(self):
        self._enabled = True
        self._initialized = False
        self._usage_stats: Dict[str, Dict[str, int]] = {}
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this cookbook (usually the cookbook directory name)."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Brief description of what this cookbook provides."""
        pass
    
    @property
    def version(self) -> str:
        """Version of the adapter."""
        return "1.0.0"
    
    @property
    def author(self) -> str:
        """Author of the cookbook."""
        return "Unknown"
    
    @property
    def tags(self) -> List[str]:
        """Tags for categorizing the cookbook."""
        return []
    
    @property
    def dependencies(self) -> List[str]:
        """List of Python package dependencies."""
        return []
    
    @property
    @abstractmethod
    def actions(self) -> List[AdapterAction]:
        """List of actions this adapter provides."""
        pass
    
    @abstractmethod
    def execute(
        self, 
        action_name: str, 
        params: Dict[str, Any], 
        shared: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute an action.
        
        Args:
            action_name: Name of the action to execute
            params: Parameters for the action
            shared: Shared state dictionary
            
        Returns:
            Dict with:
                - success (bool): Whether the action succeeded
                - result (Any): The result if successful
                - error (str): Error message if failed
                - context_update (str): Optional text to add to agent context
        """
        pass
    
    def initialize(self, shared: Dict[str, Any]) -> None:
        """
        Initialize the adapter. Called once when the adapter is first loaded.
        Override to perform setup like loading models, connecting to databases, etc.
        """
        self._initialized = True
    
    def cleanup(self, shared: Dict[str, Any]) -> None:
        """
        Cleanup resources. Called when the agent session ends.
        Override to release resources.
        """
        pass
    
    def get_action(self, name: str) -> Optional[AdapterAction]:
        """Get an action by name."""
        for action in self.actions:
            if action.name == name:
                return action
        return None
    
    @property
    def enabled(self) -> bool:
        return self._enabled
    
    def enable(self) -> None:
        self._enabled = True
    
    def disable(self) -> None:
        self._enabled = False
    
    def record_usage(self, action_name: str, success: bool) -> None:
        """Record usage statistics."""
        if action_name not in self._usage_stats:
            self._usage_stats[action_name] = {"calls": 0, "successes": 0, "failures": 0}
        
        self._usage_stats[action_name]["calls"] += 1
        if success:
            self._usage_stats[action_name]["successes"] += 1
        else:
            self._usage_stats[action_name]["failures"] += 1
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "name": self.name,
            "enabled": self._enabled,
            "initialized": self._initialized,
            "actions": self._usage_stats.copy()
        }
    
    def format_for_prompt(self) -> str:
        """Format adapter info for LLM prompt."""
        lines = [f"## {self.name}", f"{self.description}", ""]
        
        for action in self.actions:
            lines.append(action.format_for_prompt())
            lines.append("")
        
        return "\n".join(lines)
    
    def check_dependencies(self) -> tuple:
        """
        Check if all dependencies are available.
        Returns (all_available, missing_list).
        """
        missing = []
        for dep in self.dependencies:
            try:
                __import__(dep.split("[")[0])  # Handle extras like package[extra]
            except ImportError:
                missing.append(dep)
        return len(missing) == 0, missing


class AdapterRegistry:
    """
    Registry for cookbook adapters.
    
    Manages adapter lifecycle, provides action routing,
    and handles adapter discovery.
    """
    
    _instance: Optional["AdapterRegistry"] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._adapters: Dict[str, CookbookAdapter] = {}
                cls._instance._action_map: Dict[str, str] = {}  # action_name -> adapter_name
            return cls._instance
    
    def register(self, adapter: CookbookAdapter) -> None:
        """Register an adapter."""
        if not isinstance(adapter, CookbookAdapter):
            raise TypeError(f"Expected CookbookAdapter, got {type(adapter)}")
        
        with self._lock:
            # Check for action name conflicts
            for action in adapter.actions:
                full_name = f"{adapter.name}.{action.name}"
                if action.name in self._action_map:
                    existing = self._action_map[action.name]
                    if existing != adapter.name:
                        # Use fully qualified name for conflicts
                        print(f"Warning: Action '{action.name}' exists in '{existing}', "
                              f"using '{full_name}' for disambiguation")
                
                self._action_map[action.name] = adapter.name
                self._action_map[full_name] = adapter.name
            
            self._adapters[adapter.name] = adapter
    
    def unregister(self, name: str) -> bool:
        """Unregister an adapter."""
        with self._lock:
            if name in self._adapters:
                adapter = self._adapters[name]
                
                # Remove action mappings
                for action in adapter.actions:
                    if self._action_map.get(action.name) == name:
                        del self._action_map[action.name]
                    full_name = f"{name}.{action.name}"
                    if full_name in self._action_map:
                        del self._action_map[full_name]
                
                del self._adapters[name]
                return True
            return False
    
    def get(self, name: str) -> Optional[CookbookAdapter]:
        """Get an adapter by name."""
        return self._adapters.get(name)
    
    def list_adapters(self, enabled_only: bool = True) -> List[CookbookAdapter]:
        """List all registered adapters."""
        adapters = list(self._adapters.values())
        if enabled_only:
            adapters = [a for a in adapters if a.enabled]
        return adapters
    
    def find_action(self, action_name: str) -> Optional[tuple]:
        """
        Find which adapter provides an action.
        Returns (adapter, action) or None.
        """
        adapter_name = self._action_map.get(action_name)
        if not adapter_name:
            return None
        
        adapter = self._adapters.get(adapter_name)
        if not adapter or not adapter.enabled:
            return None
        
        # Handle fully qualified names
        if "." in action_name:
            _, simple_name = action_name.rsplit(".", 1)
        else:
            simple_name = action_name
        
        action = adapter.get_action(simple_name)
        if action:
            return adapter, action
        
        return None
    
    def list_all_actions(self, enabled_only: bool = True) -> List[tuple]:
        """List all available actions as (adapter_name, action) tuples."""
        actions = []
        for adapter in self.list_adapters(enabled_only=enabled_only):
            for action in adapter.actions:
                actions.append((adapter.name, action))
        return actions
    
    def execute_action(
        self,
        action_name: str,
        params: Dict[str, Any],
        shared: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute an action by name."""
        result = self.find_action(action_name)
        
        if result is None:
            return {
                "success": False,
                "error": f"Action '{action_name}' not found"
            }
        
        adapter, action = result
        
        # Validate parameters
        valid, error = action.validate_params(params)
        if not valid:
            return {"success": False, "error": error}
        
        # Initialize adapter if needed
        if not adapter._initialized:
            try:
                adapter.initialize(shared)
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Adapter initialization failed: {e}"
                }
        
        # Execute
        try:
            # Handle fully qualified names
            if "." in action_name:
                _, simple_name = action_name.rsplit(".", 1)
            else:
                simple_name = action_name
            
            exec_result = adapter.execute(simple_name, params, shared)
            adapter.record_usage(simple_name, exec_result.get("success", False))
            return exec_result
            
        except Exception as e:
            adapter.record_usage(action_name, False)
            return {
                "success": False,
                "error": f"Action execution failed: {type(e).__name__}: {e}"
            }
    
    def format_all_actions_for_prompt(self) -> str:
        """Format all available actions for LLM prompt."""
        adapters = self.list_adapters(enabled_only=True)
        
        if not adapters:
            return "No capabilities available."
        
        sections = []
        for adapter in adapters:
            sections.append(adapter.format_for_prompt())
        
        return "\n\n".join(sections)
    
    def initialize_all(self, shared: Dict[str, Any]) -> None:
        """Initialize all adapters."""
        for adapter in self._adapters.values():
            if adapter.enabled and not adapter._initialized:
                try:
                    adapter.initialize(shared)
                except Exception as e:
                    print(f"Warning: Failed to initialize '{adapter.name}': {e}")
    
    def cleanup_all(self, shared: Dict[str, Any]) -> None:
        """Cleanup all adapters."""
        for adapter in self._adapters.values():
            try:
                adapter.cleanup(shared)
            except Exception as e:
                print(f"Warning: Failed to cleanup '{adapter.name}': {e}")
    
    def get_stats(self) -> List[Dict[str, Any]]:
        """Get statistics for all adapters."""
        return [adapter.stats for adapter in self._adapters.values()]
    
    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance._adapters.clear()
                cls._instance._action_map.clear()
            cls._instance = None


def get_adapter_registry() -> AdapterRegistry:
    """Get the global adapter registry."""
    return AdapterRegistry()
