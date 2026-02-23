"""
Unified Agent Nodes

Provides the core nodes for the unified agent that can use any cookbook's capabilities.
"""

import json
import yaml
from typing import Dict, Any, List, Optional, Tuple

from pocketflow import Node

from adapters import AdapterRegistry, get_adapter_registry


# Maximum context size in characters before summarization
MAX_CONTEXT_SIZE = 500000
# Maximum number of recent actions to show in full detail
MAX_RECENT_ACTIONS = 10


def _safe_yaml_parse(response: str, fallback: Optional[Dict] = None) -> Dict[str, Any]:
    """Safely parse YAML from LLM response with robust extraction."""
    if not response:
        return fallback or {}
    
    yaml_str = None
    
    # Strategy 1: Find ```yaml ... ``` block (handle nested backticks)
    if "```yaml" in response:
        start_idx = response.find("```yaml") + 7
        # Find the closing ``` that's on its own line or at end
        remaining = response[start_idx:]
        # Look for ``` that appears at start of line or after newline
        end_idx = -1
        search_pos = 0
        while search_pos < len(remaining):
            pos = remaining.find("```", search_pos)
            if pos == -1:
                break
            # Check if this ``` is at start of line (or start of string)
            if pos == 0 or remaining[pos-1] == '\n':
                end_idx = pos
                break
            search_pos = pos + 3
        
        if end_idx > 0:
            yaml_str = remaining[:end_idx].strip()
    
    # Strategy 2: Try to find action: line and extract structured content
    if yaml_str is None and "action:" in response:
        lines = response.split('\n')
        yaml_lines = []
        in_yaml = False
        brace_count = 0
        
        for line in lines:
            stripped = line.strip()
            # Start capturing at thinking: or action:
            if stripped.startswith('thinking:') or stripped.startswith('action:'):
                in_yaml = True
            if in_yaml:
                yaml_lines.append(line)
                # Track braces for nested structures
                brace_count += line.count('{') - line.count('}')
        
        if yaml_lines:
            yaml_str = '\n'.join(yaml_lines)
    
    # Strategy 3: Fall back to first code block
    if yaml_str is None and "```" in response:
        try:
            start = response.find("```") + 3
            # Skip language identifier if present
            newline = response.find('\n', start)
            if newline != -1 and newline - start < 20:
                start = newline + 1
            end = response.find("```", start)
            if end > start:
                yaml_str = response[start:end].strip()
        except (IndexError, ValueError):
            pass
    
    if yaml_str is None:
        yaml_str = response.strip()
    
    # Try to parse YAML
    try:
        result = yaml.safe_load(yaml_str)
        if isinstance(result, dict):
            return result
    except yaml.YAMLError as e:
        # Try to extract at least action and parameters via regex
        import re
        action_match = re.search(r'action:\s*(\S+)', yaml_str)
        if action_match:
            result = {"action": action_match.group(1), "parameters": {}, "thinking": "Partial parse"}
            # Try to extract simple parameters
            param_match = re.search(r'parameters:\s*\n((?:\s+\S+:.*\n?)+)', yaml_str)
            if param_match:
                param_lines = param_match.group(1).strip().split('\n')
                for pline in param_lines:
                    pline = pline.strip()
                    if ':' in pline and not pline.startswith('#'):
                        key, _, val = pline.partition(':')
                        key = key.strip()
                        val = val.strip().strip('"\'')
                        if key and val:
                            result["parameters"][key] = val
            return result
    
    return fallback or {}


def _normalize_params(params: Dict[str, Any]) -> str:
    """Create a normalized string representation of parameters for comparison."""
    try:
        return json.dumps(params, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return str(sorted(params.items()))


def _format_action_history(action_history: List[Dict[str, Any]]) -> str:
    """Format action history for the prompt, with recent actions in detail."""
    if not action_history:
        return "No previous actions taken."
    
    lines = []
    
    # Show summary of older actions
    if len(action_history) > MAX_RECENT_ACTIONS:
        older_actions = action_history[:-MAX_RECENT_ACTIONS]
        action_counts = {}
        for entry in older_actions:
            action = entry["action"]
            action_counts[action] = action_counts.get(action, 0) + 1
        
        summary_parts = [f"{action}(x{count})" for action, count in action_counts.items()]
        lines.append(f"[Earlier actions: {', '.join(summary_parts)}]")
        lines.append("")
    
    # Show recent actions in detail
    recent_actions = action_history[-MAX_RECENT_ACTIONS:]
    for i, entry in enumerate(recent_actions, 1):
        step_num = len(action_history) - len(recent_actions) + i
        action = entry["action"]
        params_str = ", ".join(f"{k}={repr(v)[:50]}" for k, v in entry.get("params", {}).items())
        success = entry.get("success", False)
        status = "✓" if success else "✗"
        
        lines.append(f"Step {step_num}: {action}({params_str}) [{status}]")
        
        # Show result/error summary (truncated)
        result_summary = entry.get("result_summary", "")
        if result_summary:
            # Truncate long results
            if len(result_summary) > 500:
                result_summary = result_summary[:500] + "..."
            lines.append(f"  → {result_summary}")
    
    return "\n".join(lines)


def _check_duplicate_action(
    action: str,
    params: Dict[str, Any],
    action_history: List[Dict[str, Any]],
    lookback: int = 5
) -> Optional[Dict[str, Any]]:
    """Check if this exact action+params was recently attempted. Returns the previous entry if found."""
    if not action_history:
        return None
    
    normalized = _normalize_params(params)
    
    for entry in action_history[-lookback:]:
        if entry.get("action") == action and entry.get("params_normalized") == normalized:
            return entry
    
    return None


class DecideAction(Node):
    """
    Main decision node that chooses which action to take.
    
    Uses the adapter registry to present ALL available actions to the LLM.
    Maintains action history to prevent loops and provide context.
    """
    
    def __init__(self, max_iterations: int = 100):
        super().__init__()
        self.max_iterations = max_iterations
    
    def prep(self, shared: Dict[str, Any]) -> Tuple[str, str, str, str, int]:
        question = shared.get("question", "")
        
        # Get action history (structured) and format it
        action_history = shared.get("action_history", [])
        history_prompt = _format_action_history(action_history)
        
        # Get accumulated context (detailed results - may be truncated)
        context = shared.get("context", "")
        if len(context) > MAX_CONTEXT_SIZE:
            # Truncate older context, keep recent
            context = "...[earlier context truncated]...\n" + context[-MAX_CONTEXT_SIZE:]
        
        # Get available actions from registry
        registry: AdapterRegistry = shared.get("adapter_registry")
        if registry:
            actions_prompt = registry.format_all_actions_for_prompt()
        else:
            actions_prompt = "No capabilities available"
        
        iteration = shared.get("iteration", 0)
        
        return question, history_prompt, context, actions_prompt, iteration
    
    def exec(self, inputs: Tuple[str, str, str, str, int]) -> Dict[str, Any]:
        question, history_prompt, context, actions_prompt, iteration = inputs
        
        from utils.call_llm import call_llm
        
        base_prompt = f"""You are a unified AI agent with access to multiple capabilities from various cookbooks.
Choose the best action to help answer the user's question.

### USER QUESTION
{question}

### ACTION HISTORY (What you've already done)
{history_prompt}

### DETAILED CONTEXT (Results from previous actions)
{context if context else "No detailed context yet."}

### AVAILABLE ACTIONS
{actions_prompt}

### SPECIAL ACTIONS
[answer]
  Description: Provide the final answer to the user's question
  Parameters:
    - answer (str, required): The complete answer

### CRITICAL RULES
1. DO NOT repeat an action with the same parameters if it already succeeded - use the results you have
2. DO NOT keep reading the same file repeatedly - you already have its contents in the context
3. If you have enough information to answer, use the 'answer' action immediately
4. If a previous action failed, try a DIFFERENT approach
5. Break complex tasks into steps, but don't repeat completed steps

### RESPONSE FORMAT (STRICT - follow exactly)
You MUST respond with ONLY a YAML code block in this EXACT format:
```yaml
thinking: "Brief one-line reasoning"
action: action_name
parameters:
  param1: value1
  param2: value2
```

IMPORTANT FORMAT RULES:
- Use "quotes" for string values, especially multi-line content
- For file content, use a SINGLE line with \\n for newlines, e.g.: content: "line1\\nline2\\nline3"
- Do NOT use YAML multi-line syntax (| or >)
- Keep parameter values simple and on one line each
- The YAML block must be parseable - no nested code blocks inside"""
        
        # Try up to 2 times to get valid YAML
        for parse_attempt in range(2):
            if parse_attempt > 0:
                response = call_llm(base_prompt + "\n\nRETRY: Your previous response could not be parsed. Please respond with SIMPLE YAML only.")
            else:
                response = call_llm(base_prompt)
            
            print(f"LLM Response (attempt {parse_attempt + 1}):\n{response[:500]}...")
            
            decision = _safe_yaml_parse(response, None)
            
            if decision is not None and "action" in decision:
                if "thinking" not in decision:
                    decision["thinking"] = ""
                if "parameters" not in decision:
                    decision["parameters"] = {}
                
                return decision
        
        # All parse attempts failed - return fallback
        print(f"WARNING: Failed to get valid response")
        return {
            "action": "answer",
            "parameters": {"answer": f"I encountered issues processing this request. Based on what I know: {context[-1000:] if context else 'No context available'}"},
            "thinking": "Failed to get valid response"
        }
    
    def post(self, shared: Dict[str, Any], prep_res: Any, exec_res: Dict[str, Any]) -> str:
        _, _, _, _, iteration = prep_res
        
        action = exec_res.get("action", "answer")
        parameters = exec_res.get("parameters", {})
        thinking = exec_res.get("thinking", "")
        
        # Update iteration count
        shared["iteration"] = iteration + 1
        
        # Initialize action history if needed
        if "action_history" not in shared:
            shared["action_history"] = []
        
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
    
    def prep(self, shared: Dict[str, Any]) -> Tuple[str, Dict[str, Any], AdapterRegistry, Dict[str, Any]]:
        action = shared.get("pending_action", "")
        parameters = shared.get("pending_parameters", {})
        registry = shared.get("adapter_registry")
        
        return action, parameters, registry, shared
    
    def exec(self, inputs: Tuple[str, Dict[str, Any], AdapterRegistry, Dict[str, Any]]) -> Dict[str, Any]:
        action, parameters, registry, shared = inputs
        
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
        result = registry.execute_action(action, parameters, shared=shared)
        result["is_final"] = False
        
        return result
    
    def post(self, shared: Dict[str, Any], prep_res: Any, exec_res: Dict[str, Any]) -> str:
        action, parameters, registry, shared = prep_res
        
        # Check if this is the final answer
        if exec_res.get("is_final"):
            shared["answer"] = exec_res.get("result", "")
            print(f"✅ Final answer ready")
            return "done"
        
        # Initialize action history if not present
        if "action_history" not in shared:
            shared["action_history"] = []
        
        # Check for duplicate action BEFORE adding to history (for loop reminder)
        duplicate = _check_duplicate_action(action, parameters, shared["action_history"])
        
        # Create a summary of the result for action history
        if exec_res.get("success"):
            context_update = exec_res.get("context_update", str(exec_res.get("result", "")))
            # Handle empty or None results
            if not context_update or context_update in ("", "None", "null"):
                context_update = "(The tool call returned no output)"
            # Create a shorter summary for history
            result_summary = context_update[:300] if len(context_update) > 300 else context_update
            status_msg = "Success"
        else:
            error = exec_res.get("error", "")
            # Handle empty error messages
            if not error:
                error = "(The tool call returned no error message)"
            result_summary = f"Error: {error}"
            status_msg = f"Failed: {error[:80]}"
        
        # Record in action history (structured, for loop detection)
        history_entry = {
            "action": action,
            "params": parameters,
            "params_normalized": _normalize_params(parameters),
            "success": exec_res.get("success", False),
            "result_summary": result_summary
        }
        shared["action_history"].append(history_entry)
        
        # Update detailed context (full results)
        prev_context = shared.get("context", "")
        
        if exec_res.get("success"):
            context_update = exec_res.get("context_update", str(exec_res.get("result", "")))
            # Handle empty or None results
            if not context_update or context_update in ("", "None", "null"):
                context_update = "(The tool call returned no output)"
            new_context = f"\n\n[Action: {action}]\n{context_update}"
            print(f"   ✓ Success")
        else:
            error = exec_res.get("error", "")
            # Handle empty error messages
            if not error:
                error = "(The tool call returned no error message)"
            new_context = f"\n\n[Action: {action}]\nFailed: {error}"
            print(f"   ✗ Failed: {error[:80]}")
        
        # Add loop reminder if this action was done before (one reminder per execution)
        if duplicate:
            prev_result = duplicate.get("result_summary", "")[:200]
            loop_reminder = f"\n\n⚠️ NOTE: You have performed this exact action before with result: '{prev_result}...'. Consider using the existing result or trying a different approach."
            new_context += loop_reminder
            print(f"   ⚠️ Duplicate action detected (informing agent)")
        
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
