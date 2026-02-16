"""
Adapter for pocketflow-thinking cookbook.

Provides chain-of-thought reasoning capabilities.
"""

import sys
import os
from typing import Dict, Any, List

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pocketflow-unified-agent'))

from adapters.base import CookbookAdapter, AdapterAction


class ThinkingAdapter(CookbookAdapter):
    """Adapter for the thinking/chain-of-thought cookbook."""
    
    def __init__(self):
        super().__init__()
        self._thoughts = []
    
    @property
    def name(self) -> str:
        return "pocketflow-thinking"
    
    @property
    def description(self) -> str:
        return "Chain-of-thought reasoning with planning and step-by-step execution"
    
    @property
    def tags(self) -> List[str]:
        return ["thinking", "reasoning", "planning"]
    
    @property
    def dependencies(self) -> List[str]:
        return ["openai", "pyyaml"]
    
    @property
    def actions(self) -> List[AdapterAction]:
        return [
            AdapterAction(
                name="think_through",
                description="Think through a problem step by step using chain-of-thought reasoning",
                parameters={
                    "problem": {
                        "type": "str",
                        "description": "The problem to think through",
                        "required": True
                    },
                    "max_steps": {
                        "type": "int",
                        "description": "Maximum thinking steps",
                        "required": False,
                        "default": 5
                    }
                }
            ),
            AdapterAction(
                name="plan_and_solve",
                description="Create a plan and solve a problem methodically",
                parameters={
                    "problem": {
                        "type": "str",
                        "description": "Problem to solve",
                        "required": True
                    }
                }
            )
        ]
    
    def execute(
        self,
        action_name: str,
        params: Dict[str, Any],
        shared: Dict[str, Any]
    ) -> Dict[str, Any]:
        if action_name == "think_through":
            return self._think_through(params, shared)
        elif action_name == "plan_and_solve":
            return self._plan_and_solve(params, shared)
        
        return {"success": False, "error": f"Unknown action: {action_name}"}
    
    def _think_through(self, params: Dict[str, Any], shared: Dict[str, Any]) -> Dict[str, Any]:
        """Execute chain of thought reasoning."""
        problem = params.get("problem", "")
        max_steps = params.get("max_steps", 5)
        
        if not problem:
            return {"success": False, "error": "Problem cannot be empty"}
        
        try:
            from flow import create_thinking_flow
            
            flow = create_thinking_flow()
            flow_shared = {"problem": problem}
            flow.run(flow_shared)
            
            solution = flow_shared.get("solution", "No solution found")
            thoughts = flow_shared.get("thoughts", [])
            
            # Format thoughts
            thought_summary = []
            for t in thoughts[:max_steps]:
                num = t.get("thought_number", "?")
                thinking = t.get("current_thinking", "")[:200]
                thought_summary.append(f"Step {num}: {thinking}...")
            
            context = f"Thinking through: {problem}\n\n"
            context += "\n".join(thought_summary)
            context += f"\n\nSolution: {solution}"
            
            return {
                "success": True,
                "result": solution,
                "context_update": context
            }
            
        except ImportError:
            # Fallback: simple thinking prompt
            return self._simple_thinking(problem)
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _simple_thinking(self, problem: str) -> Dict[str, Any]:
        """Fallback simple thinking."""
        try:
            # Try to import call_llm from various places
            try:
                from utils import call_llm
            except ImportError:
                from utils.call_llm import call_llm
            
            prompt = f"""Think through this problem step by step:

Problem: {problem}

Let's solve this step by step:
1. First, let me understand what we're asked to do.
2. Then, I'll break down the problem.
3. Finally, I'll provide a solution.

Thinking:"""
            
            response = call_llm(prompt)
            
            return {
                "success": True,
                "result": response,
                "context_update": f"Thought through: {problem}\n\n{response}"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _plan_and_solve(self, params: Dict[str, Any], shared: Dict[str, Any]) -> Dict[str, Any]:
        """Create a plan and solve."""
        problem = params.get("problem", "")
        
        if not problem:
            return {"success": False, "error": "Problem cannot be empty"}
        
        try:
            try:
                from utils import call_llm
            except ImportError:
                from utils.call_llm import call_llm
            
            prompt = f"""Create a plan to solve this problem, then execute it:

Problem: {problem}

First, create a numbered plan:
1. ...
2. ...

Then execute each step and provide the final answer.

Plan and Solution:"""
            
            response = call_llm(prompt)
            
            return {
                "success": True,
                "result": response,
                "context_update": f"Plan and solve: {problem}\n\n{response}"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


def get_adapter() -> CookbookAdapter:
    return ThinkingAdapter()
