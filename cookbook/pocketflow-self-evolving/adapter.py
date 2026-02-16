"""
Adapter for pocketflow-self-evolving cookbook.

Provides tool creation and management capabilities.
"""

import sys
import os
from typing import Dict, Any, List

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pocketflow-unified-agent'))

from adapters.base import CookbookAdapter, AdapterAction


class SelfEvolvingAdapter(CookbookAdapter):
    """Adapter for the self-evolving tool creation cookbook."""
    
    def __init__(self):
        super().__init__()
        self._registry = None
    
    @property
    def name(self) -> str:
        return "pocketflow-self-evolving"
    
    @property
    def description(self) -> str:
        return "Create, register, search, and execute custom reusable tools"
    
    @property
    def tags(self) -> List[str]:
        return ["tool", "self-evolving", "agent"]
    
    @property
    def dependencies(self) -> List[str]:
        return ["openai", "pyyaml"]
    
    @property
    def actions(self) -> List[AdapterAction]:
        return [
            AdapterAction(
                name="create_tool",
                description="Create and register a new reusable tool",
                parameters={
                    "name": {"type": "str", "description": "Tool name", "required": True},
                    "description": {"type": "str", "description": "What the tool does", "required": True},
                    "parameters_hint": {"type": "str", "description": "Hint about parameters", "required": False, "default": ""}
                }
            ),
            AdapterAction(
                name="search_tools",
                description="Search for existing tools",
                parameters={
                    "query": {"type": "str", "description": "Search query", "required": True}
                }
            ),
            AdapterAction(
                name="use_tool",
                description="Execute a registered tool",
                parameters={
                    "tool_name": {"type": "str", "description": "Name of tool to execute", "required": True},
                    "inputs": {"type": "dict", "description": "Tool inputs", "required": False, "default": {}}
                }
            ),
            AdapterAction(
                name="list_tools",
                description="List all registered tools",
                parameters={}
            )
        ]
    
    def initialize(self, shared: Dict[str, Any]) -> None:
        try:
            from tools.tool_registry import ToolRegistry, get_registry
            db_path = shared.get("tool_registry_path", "tool_registry.db")
            self._registry = get_registry(db_path=db_path)
        except ImportError:
            self._registry = None
        self._initialized = True
    
    def execute(
        self,
        action_name: str,
        params: Dict[str, Any],
        shared: Dict[str, Any]
    ) -> Dict[str, Any]:
        if self._registry is None:
            self.initialize(shared)
        
        if action_name == "create_tool":
            return self._create_tool(params, shared)
        elif action_name == "search_tools":
            return self._search_tools(params)
        elif action_name == "use_tool":
            return self._use_tool(params)
        elif action_name == "list_tools":
            return self._list_tools()
        
        return {"success": False, "error": f"Unknown action: {action_name}"}
    
    def _create_tool(self, params: Dict[str, Any], shared: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new tool."""
        name = params.get("name", "")
        description = params.get("description", "")
        hint = params.get("parameters_hint", "")
        
        if not name or not description:
            return {"success": False, "error": "Name and description required"}
        
        try:
            from nodes import CreateTool
            from tools.tool_registry import ToolRegistry
            
            node = CreateTool()
            node_shared = {
                "new_tool_name": name,
                "new_tool_description": description,
                "question": hint or description,
                "context": "",
                "tool_registry": self._registry
            }
            
            prep_res = node.prep(node_shared)
            exec_res = node.exec(prep_res)
            node.post(node_shared, prep_res, exec_res)
            
            if exec_res.get("source_code"):
                return {
                    "success": True,
                    "result": f"Created tool: {name}",
                    "context_update": f"Tool '{name}' created: {description}"
                }
            else:
                return {"success": False, "error": exec_res.get("error", "Failed to create")}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _search_tools(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Search tools."""
        query = params.get("query", "")
        
        if not query:
            return {"success": False, "error": "Query required"}
        
        if not self._registry:
            return {"success": False, "error": "Registry not initialized"}
        
        try:
            results = self._registry.search(query, top_k=5)
            
            if results:
                context = f"Found {len(results)} tools for '{query}':\n"
                tool_list = []
                for tool, score in results:
                    name = tool.metadata.name
                    desc = tool.metadata.description
                    context += f"- {name}: {desc[:80]}...\n"
                    tool_list.append({"name": name, "description": desc, "score": score})
                
                return {
                    "success": True,
                    "result": tool_list,
                    "context_update": context
                }
            else:
                return {
                    "success": True,
                    "result": [],
                    "context_update": f"No tools found for '{query}'"
                }
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _use_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool."""
        tool_name = params.get("tool_name", "")
        inputs = params.get("inputs", {})
        
        if not tool_name:
            return {"success": False, "error": "Tool name required"}
        
        if not self._registry:
            return {"success": False, "error": "Registry not initialized"}
        
        try:
            result = self._registry.execute(tool_name, inputs)
            return {
                "success": True,
                "result": result,
                "context_update": f"Tool '{tool_name}' result: {result}"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _list_tools(self) -> Dict[str, Any]:
        """List all tools."""
        if not self._registry:
            return {"success": False, "error": "Registry not initialized"}
        
        try:
            tools = self._registry.list_tools(enabled_only=True)
            
            if tools:
                context = f"Registered tools ({len(tools)}):\n"
                tool_list = []
                for tool in tools:
                    name = tool.metadata.name
                    desc = tool.metadata.description
                    context += f"- {name}: {desc[:60]}...\n"
                    tool_list.append({"name": name, "description": desc})
                
                return {
                    "success": True,
                    "result": tool_list,
                    "context_update": context
                }
            else:
                return {
                    "success": True,
                    "result": [],
                    "context_update": "No tools registered"
                }
                
        except Exception as e:
            return {"success": False, "error": str(e)}


def get_adapter() -> CookbookAdapter:
    return SelfEvolvingAdapter()
