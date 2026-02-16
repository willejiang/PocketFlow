"""
Template for creating a cookbook adapter.

Copy this file to your cookbook directory as 'adapter.py' and customize it.
"""

from typing import Dict, Any, List

# Import from the unified agent's adapters module
# When placed in a cookbook directory, this import path works:
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pocketflow-unified-agent'))

from adapters.base import CookbookAdapter, AdapterAction


class MyCookbookAdapter(CookbookAdapter):
    """
    Adapter for my-cookbook.
    
    Replace this with a description of what your cookbook does.
    """
    
    @property
    def name(self) -> str:
        """Return the cookbook name (usually the directory name)."""
        return "pocketflow-my-cookbook"
    
    @property
    def description(self) -> str:
        """Brief description of what this cookbook provides."""
        return "Description of my cookbook's capabilities"
    
    @property
    def version(self) -> str:
        """Version of the adapter."""
        return "1.0.0"
    
    @property
    def tags(self) -> List[str]:
        """Tags for categorizing the cookbook."""
        return ["agent", "tool"]  # Add relevant tags
    
    @property
    def dependencies(self) -> List[str]:
        """Python package dependencies."""
        return ["openai"]  # List required packages
    
    @property
    def actions(self) -> List[AdapterAction]:
        """
        Define the actions this adapter provides.
        
        Each action should correspond to a capability of your cookbook.
        """
        return [
            AdapterAction(
                name="my_action",
                description="Description of what this action does",
                parameters={
                    "input": {
                        "type": "str",
                        "description": "The input parameter",
                        "required": True
                    },
                    "option": {
                        "type": "int",
                        "description": "An optional parameter",
                        "required": False,
                        "default": 10
                    }
                }
            ),
            # Add more actions as needed
        ]
    
    def initialize(self, shared: Dict[str, Any]) -> None:
        """
        Initialize the adapter.
        
        Called once when the adapter is first loaded.
        Use this to load models, connect to databases, etc.
        """
        # Example: Load your cookbook's modules
        # from . import nodes, flow
        # self._flow = flow.create_flow()
        pass
    
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
            params: Parameters passed to the action
            shared: Shared state dictionary
            
        Returns:
            Dict with:
                - success (bool): Whether the action succeeded
                - result (Any): The result if successful
                - error (str): Error message if failed
                - context_update (str): Text to add to agent's context
        """
        if action_name == "my_action":
            return self._execute_my_action(params, shared)
        
        return {
            "success": False,
            "error": f"Unknown action: {action_name}"
        }
    
    def _execute_my_action(
        self,
        params: Dict[str, Any],
        shared: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute the my_action action."""
        try:
            input_value = params.get("input", "")
            option = params.get("option", 10)
            
            # Your action logic here
            # Example: Run your cookbook's flow
            # result = self._flow.run({"input": input_value})
            result = f"Processed: {input_value} with option {option}"
            
            return {
                "success": True,
                "result": result,
                "context_update": f"My action completed: {result}"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def cleanup(self, shared: Dict[str, Any]) -> None:
        """
        Cleanup resources.
        
        Called when the agent session ends.
        """
        pass


# This function is called by the adapter discovery system
def get_adapter() -> CookbookAdapter:
    """Return an instance of the adapter."""
    return MyCookbookAdapter()
