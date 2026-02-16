"""
Adapter for pocketflow-tao cookbook.

Provides Thought-Action-Observation reasoning.
"""

import sys
import os
from typing import Dict, Any, List

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pocketflow-unified-agent'))

from adapters.base import CookbookAdapter, AdapterAction


class TAOAdapter(CookbookAdapter):
    """Adapter for the TAO (Thought-Action-Observation) cookbook."""
    
    def __init__(self):
        super().__init__()
        self._thoughts = []
        self._observations = []
    
    @property
    def name(self) -> str:
        return "pocketflow-tao"
    
    @property
    def description(self) -> str:
        return "Thought-Action-Observation reasoning loop for problem solving"
    
    @property
    def tags(self) -> List[str]:
        return ["thinking", "reasoning", "agent"]
    
    @property
    def dependencies(self) -> List[str]:
        return ["openai", "pyyaml"]
    
    @property
    def actions(self) -> List[AdapterAction]:
        return [
            AdapterAction(
                name="tao_solve",
                description="Solve a problem using Thought-Action-Observation loop",
                parameters={
                    "query": {"type": "str", "description": "Problem to solve", "required": True},
                    "max_iterations": {"type": "int", "description": "Max TAO cycles", "required": False, "default": 5}
                }
            ),
            AdapterAction(
                name="tao_think",
                description="Execute one thinking step",
                parameters={
                    "query": {"type": "str", "description": "What to think about", "required": True}
                }
            )
        ]
    
    def execute(
        self,
        action_name: str,
        params: Dict[str, Any],
        shared: Dict[str, Any]
    ) -> Dict[str, Any]:
        if action_name == "tao_solve":
            return self._solve(params, shared)
        elif action_name == "tao_think":
            return self._think(params, shared)
        
        return {"success": False, "error": f"Unknown action: {action_name}"}
    
    def _solve(self, params: Dict[str, Any], shared: Dict[str, Any]) -> Dict[str, Any]:
        """Run full TAO loop."""
        query = params.get("query", "")
        max_iterations = params.get("max_iterations", 5)
        
        if not query:
            return {"success": False, "error": "Query required"}
        
        try:
            from flow import create_tao_flow
            
            flow = create_tao_flow()
            flow_shared = {"query": query}
            flow.run(flow_shared)
            
            answer = flow_shared.get("final_answer", "No answer found")
            thoughts = flow_shared.get("thoughts", [])
            
            context = f"TAO solving: {query}\n\n"
            for t in thoughts:
                context += f"Thought: {t.get('thinking', '')[:100]}...\n"
            context += f"\nAnswer: {answer}"
            
            return {
                "success": True,
                "result": answer,
                "context_update": context
            }
            
        except ImportError:
            # Fallback: simple reasoning
            return self._simple_tao(query)
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _think(self, params: Dict[str, Any], shared: Dict[str, Any]) -> Dict[str, Any]:
        """Execute one thinking step."""
        query = params.get("query", "")
        
        if not query:
            return {"success": False, "error": "Query required"}
        
        return self._simple_tao(query)
    
    def _simple_tao(self, query: str) -> Dict[str, Any]:
        """Simple TAO reasoning."""
        try:
            try:
                from utils import call_llm
            except ImportError:
                from utils.call_llm import call_llm
            
            import yaml
            
            prompt = f"""You are an AI using Thought-Action-Observation reasoning.

Query: {query}

Think about this problem step by step:
1. THOUGHT: What do I need to figure out?
2. ACTION: What action should I take?
3. OBSERVATION: What did I learn?

Respond in YAML:
```yaml
thinking: |
    <your reasoning process>
action: <action to take>
action_input: <input for the action>
is_final: <true if this is the final answer>
```"""
            
            response = call_llm(prompt)
            
            # Parse
            if "```yaml" in response:
                yaml_str = response.split("```yaml")[1].split("```")[0].strip()
                result = yaml.safe_load(yaml_str)
            else:
                result = {"thinking": response, "is_final": True, "action_input": response}
            
            is_final = result.get("is_final", False)
            answer = result.get("action_input", result.get("thinking", ""))
            
            return {
                "success": True,
                "result": answer if is_final else result,
                "context_update": f"TAO: {query}\nThinking: {result.get('thinking', '')[:200]}..."
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


def get_adapter() -> CookbookAdapter:
    return TAOAdapter()
