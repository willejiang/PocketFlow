"""
Adapter for pocketflow-code-generator cookbook.

Provides code generation and execution capabilities.
"""

import sys
import os
from typing import Dict, Any, List

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pocketflow-unified-agent'))

from adapters.base import CookbookAdapter, AdapterAction


class CodeGeneratorAdapter(CookbookAdapter):
    """Adapter for the code generator cookbook."""
    
    @property
    def name(self) -> str:
        return "pocketflow-code-generator"
    
    @property
    def description(self) -> str:
        return "Generate and test Python code with automatic revision"
    
    @property
    def tags(self) -> List[str]:
        return ["code", "generator", "testing"]
    
    @property
    def dependencies(self) -> List[str]:
        return ["openai", "pyyaml"]
    
    @property
    def actions(self) -> List[AdapterAction]:
        return [
            AdapterAction(
                name="generate_code",
                description="Generate Python code to solve a problem with test cases",
                parameters={
                    "problem": {
                        "type": "str",
                        "description": "Description of the coding problem to solve",
                        "required": True
                    },
                    "max_iterations": {
                        "type": "int",
                        "description": "Maximum revision iterations",
                        "required": False,
                        "default": 5
                    }
                }
            ),
            AdapterAction(
                name="execute_code",
                description="Execute Python code safely",
                parameters={
                    "code": {
                        "type": "str",
                        "description": "Python code to execute",
                        "required": True
                    },
                    "inputs": {
                        "type": "dict",
                        "description": "Input parameters for the code",
                        "required": False,
                        "default": {}
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
        if action_name == "generate_code":
            return self._generate_code(params, shared)
        elif action_name == "execute_code":
            return self._execute_code(params)
        
        return {"success": False, "error": f"Unknown action: {action_name}"}
    
    def _generate_code(self, params: Dict[str, Any], shared: Dict[str, Any]) -> Dict[str, Any]:
        """Generate code using the flow."""
        problem = params.get("problem", "")
        max_iterations = params.get("max_iterations", 5)
        
        if not problem:
            return {"success": False, "error": "Problem cannot be empty"}
        
        try:
            from flow import create_code_generator_flow
            
            flow = create_code_generator_flow()
            flow_shared = {
                "problem": problem,
                "max_iterations": max_iterations
            }
            flow.run(flow_shared)
            
            code = flow_shared.get("function_code", "")
            test_results = flow_shared.get("test_results", [])
            
            passed = sum(1 for r in test_results if r.get("passed", False))
            total = len(test_results)
            
            return {
                "success": True,
                "result": {
                    "code": code,
                    "tests_passed": f"{passed}/{total}"
                },
                "context_update": f"Generated code for: {problem}\n\n```python\n{code}\n```\n\nTests: {passed}/{total} passed"
            }
            
        except ImportError:
            # Fallback: simple code generation
            return self._simple_generate(problem)
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _simple_generate(self, problem: str) -> Dict[str, Any]:
        """Fallback simple code generation."""
        try:
            try:
                from utils.call_llm import call_llm
            except ImportError:
                from utils import call_llm
            
            prompt = f"""Generate Python code to solve this problem:

{problem}

Provide a function called 'solve' that implements the solution.

```python
def solve(...):
    ...
```"""
            
            response = call_llm(prompt)
            
            # Extract code
            if "```python" in response:
                code = response.split("```python")[1].split("```")[0].strip()
            else:
                code = response
            
            return {
                "success": True,
                "result": {"code": code},
                "context_update": f"Generated code:\n```python\n{code}\n```"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _execute_code(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute code safely."""
        code = params.get("code", "")
        inputs = params.get("inputs", {})
        
        if not code:
            return {"success": False, "error": "Code cannot be empty"}
        
        try:
            from utils.code_executor import execute_python
            
            result, error = execute_python(code, inputs)
            
            if error:
                return {
                    "success": False,
                    "error": error,
                    "context_update": f"Code execution failed: {error}"
                }
            
            return {
                "success": True,
                "result": result,
                "context_update": f"Code executed. Result: {result}"
            }
            
        except ImportError:
            # Fallback: execute directly (less safe)
            return self._unsafe_execute(code, inputs)
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _unsafe_execute(self, code: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback execution (less safe)."""
        try:
            namespace = {}
            exec(code, namespace)
            
            # Try to find and call a function
            for name in ["solve", "run_code", "main"]:
                if name in namespace:
                    if inputs:
                        result = namespace[name](**inputs)
                    else:
                        result = namespace[name]()
                    return {
                        "success": True,
                        "result": result,
                        "context_update": f"Executed. Result: {result}"
                    }
            
            return {
                "success": True,
                "result": "Code executed (no return value)",
                "context_update": "Code executed"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


def get_adapter() -> CookbookAdapter:
    return CodeGeneratorAdapter()
