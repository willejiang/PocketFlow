"""
Unified Agent Nodes

Provides the core nodes for the unified agent that can use any cookbook's capabilities.
"""

import yaml
from typing import Dict, Any, List, Optional, Tuple

from pocketflow import Node

from adapters import AdapterRegistry, get_adapter_registry


def _safe_yaml_parse(response: str, fallback: Optional[Dict] = None) -> Dict[str, Any]:
    """Safely parse YAML from LLM response."""
    if not response:
        return fallback or {}
    
    yaml_str = None
    
    if "```yaml" in response:
        try:
            yaml_str = response.split("```yaml")[1].split("```")[0].strip()
        except IndexError:
            pass
    elif "```" in response:
        try:
            yaml_str = response.split("```")[1].split("```")[0].strip()
        except IndexError:
            pass
    
    if yaml_str is None:
        yaml_str = response.strip()
    
    try:
        result = yaml.safe_load(yaml_str)
        if isinstance(result, dict):
            return result
    except yaml.YAMLError:
        pass
    
    return fallback or {}


class DecideAction(Node):
    """
    Main decision node that chooses which action to take.
    
    Uses the adapter registry to present ALL available actions to the LLM.
    """
    
    def __init__(self, max_iterations: int = 100):
        super().__init__()
        self.max_iterations = max_iterations
    
    def prep(self, shared: Dict[str, Any]) -> Tuple[str, str, str, int]:
        question = shared.get("question", "")
        context = shared.get("context", "No previous actions")
        
        # Get available actions from registry
        registry: AdapterRegistry = shared.get("adapter_registry")
        if registry:
            actions_prompt = registry.format_all_actions_for_prompt()
        else:
            actions_prompt = "No capabilities available"
        
        iteration = shared.get("iteration", 0)
        
        return question, context, actions_prompt, iteration
    
    def exec(self, inputs: Tuple[str, str, str, int]) -> Dict[str, Any]:
        question, context, actions_prompt, iteration = inputs
        
        from utils.call_llm import call_llm
        
        prompt = f"""You are a unified AI agent with access to multiple capabilities from various cookbooks.
Choose the best action to help answer the user's question.

### USER QUESTION
{question}

### CONTEXT (Previous Actions & Results)
{context}

### AVAILABLE ACTIONS
{actions_prompt}

### SPECIAL ACTIONS
[answer]
  Description: Provide the final answer to the user's question
  Parameters:
    - answer (str, required): The complete answer

### INSTRUCTIONS
1. Review the question and available context
2. Choose ONE action that best helps answer the question
3. If you have enough information, use 'answer' to provide the final response
4. For complex problems, break them down into steps using available actions
5. Action names may include cookbook prefix (e.g., 'rag_retrieve_context')

### YOUR RESPONSE
Respond with YAML only:
```yaml
thinking: |
    <your reasoning about what to do>
action: <action_name>
parameters:
    <param_name>: <param_value>
```"""
        
        response = call_llm(prompt)
        print(f"LLM Response:\n{response}")
        decision = _safe_yaml_parse(response, {
            "action": "answer",
            "parameters": {"answer": "Unable to process request"},
            "thinking": "Parse failed"
        })
        
        return decision
    
    def post(self, shared: Dict[str, Any], prep_res: Any, exec_res: Dict[str, Any]) -> str:
        _, _, _, iteration = prep_res
        
        action = exec_res.get("action", "answer")
        parameters = exec_res.get("parameters", {})
        thinking = exec_res.get("thinking", "")
        
        # Update iteration count
        shared["iteration"] = iteration + 1
        
        # Check max iterations
        if shared["iteration"] >= self.max_iterations:
            print(f"⚠️ Max iterations ({self.max_iterations}) reached")
            shared["pending_action"] = "answer"
            shared["pending_parameters"] = {
                "answer": "Maximum iterations reached. " + shared.get("context", "")[-500:]
            }
            return "execute"
        
        # Store pending action
        shared["pending_action"] = action
        shared["pending_parameters"] = parameters
        shared["last_thinking"] = thinking
        
        print(f"🤔 [{iteration + 1}] {thinking[:80]}...")
        print(f"   → Action: {action}")
        
        return "execute"


class ExecuteAction(Node):
    """Execute the chosen action using the adapter registry."""
    
    def prep(self, shared: Dict[str, Any]) -> Tuple[str, Dict[str, Any], AdapterRegistry]:
        action = shared.get("pending_action", "")
        parameters = shared.get("pending_parameters", {})
        registry = shared.get("adapter_registry")
        
        return action, parameters, registry
    
    def exec(self, inputs: Tuple[str, Dict[str, Any], AdapterRegistry]) -> Dict[str, Any]:
        action, parameters, registry = inputs
        
        # Handle built-in answer action
        if action == "answer":
            return {
                "success": True,
                "result": parameters.get("answer", "No answer provided"),
                "is_final": True
            }
        
        if registry is None:
            return {
                "success": False,
                "error": "Adapter registry not initialized"
            }
        
        # Execute via registry
        result = registry.execute_action(action, parameters, {})
        result["is_final"] = False
        
        return result
    
    def post(self, shared: Dict[str, Any], prep_res: Any, exec_res: Dict[str, Any]) -> str:
        action, parameters, _ = prep_res
        
        # Check if this is the final answer
        if exec_res.get("is_final"):
            shared["answer"] = exec_res.get("result", "")
            print(f"✅ Final answer ready")
            return "done"
        
        # Update context with results
        prev_context = shared.get("context", "")
        
        if exec_res.get("success"):
            context_update = exec_res.get("context_update", str(exec_res.get("result", "")))
            new_context = f"\n\n[Action: {action}]\n{context_update}"
            print(f"   ✓ Success")
        else:
            error = exec_res.get("error", "Unknown error")
            new_context = f"\n\n[Action: {action}]\nFailed: {error}"
            print(f"   ✗ Failed: {error[:80]}")
        
        shared["context"] = prev_context + new_context
        
        # Continue decision loop
        return "decide"


class FinalizeAnswer(Node):
    """Finalize and format the answer."""
    
    def prep(self, shared: Dict[str, Any]) -> Tuple[str, str, str]:
        question = shared.get("question", "")
        context = shared.get("context", "")
        answer = shared.get("answer", "")
        
        return question, context, answer
    
    def exec(self, inputs: Tuple[str, str, str]) -> str:
        question, context, answer = inputs
        
        # If answer is already good, return it
        if answer and len(answer) > 50:
            return answer
        
        # Generate a better answer from context
        try:
            from utils.call_llm import call_llm
            
            prompt = f"""Based on the following context, provide a comprehensive answer.

Question: {question}

Context:
{context[-3000:]}

Provide a clear, well-formatted answer:"""
            
            return call_llm(prompt)
        except Exception:
            return answer or "Unable to generate answer"
    
    def post(self, shared: Dict[str, Any], prep_res: Any, exec_res: str) -> str:
        shared["final_answer"] = exec_res
        
        # Cleanup adapters
        registry = shared.get("adapter_registry")
        if registry:
            registry.cleanup_all(shared)
        
        return "done"
