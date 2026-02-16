"""
Self-Evolving Agent Nodes

Provides nodes for:
- Deciding actions including tool creation
- Creating and registering new tools
- Searching and retrieving tools
- Executing registered tools
- Answering questions using tools
"""

import yaml
import json
import re
import traceback
from typing import Optional, Dict, Any, List, Tuple

from pocketflow import Node

from tools.tool_registry import (
    ToolRegistry, 
    RegisteredTool, 
    ToolNotFoundError,
    ToolExecutionError,
    ToolValidationError,
    get_registry
)
from utils.call_llm import call_llm, get_embedding, LLMError


def _safe_yaml_parse(response: str, fallback: Optional[Dict] = None) -> Dict[str, Any]:
    """Safely parse YAML from LLM response with multiple fallback strategies."""
    if not response:
        return fallback or {}
    
    # Try to extract YAML block
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
    
    # Try parsing
    try:
        result = yaml.safe_load(yaml_str)
        if isinstance(result, dict):
            return result
    except yaml.YAMLError:
        pass
    
    # Fallback: try to extract key-value pairs
    try:
        result = {}
        for line in yaml_str.split("\n"):
            if ":" in line:
                key, _, value = line.partition(":")
                key = key.strip().rstrip(":")
                value = value.strip()
                if key and value:
                    result[key] = value
        if result:
            return result
    except Exception:
        pass
    
    return fallback or {}


def _get_tool_registry(shared: Dict[str, Any]) -> ToolRegistry:
    """Get or create tool registry from shared state."""
    if "tool_registry" not in shared:
        db_path = shared.get("tool_registry_path", "tool_registry.db")
        
        # Try to use embeddings if available
        embedding_fn = None
        try:
            embedding_fn = get_embedding
        except Exception:
            pass
        
        shared["tool_registry"] = get_registry(
            db_path=db_path,
            embedding_fn=embedding_fn
        )
    
    return shared["tool_registry"]


class DecideAction(Node):
    """
    Decision node that chooses between:
    - search_tools: Find relevant existing tools
    - create_tool: Create a new reusable tool
    - use_tool: Execute an existing tool
    - answer: Provide final answer
    """
    
    def prep(self, shared: Dict[str, Any]) -> Tuple[str, str, List[str], Optional[str]]:
        question = shared.get("question", "")
        context = shared.get("context", "No previous actions")
        
        # Get available tools summary
        registry = _get_tool_registry(shared)
        tools = registry.list_tools(enabled_only=True, limit=20)
        
        available_tools = []
        for tool in tools:
            sig = registry.get_tool_signature(tool.metadata.name)
            available_tools.append(f"  - {sig}: {tool.metadata.description[:100]}")
        
        tools_summary = "\n".join(available_tools) if available_tools else "  (no tools registered yet)"
        
        # Get pending tool if any
        pending_tool = shared.get("pending_tool_name")
        
        return question, context, tools_summary, pending_tool
    
    def exec(self, inputs: Tuple[str, str, str, Optional[str]]) -> Dict[str, Any]:
        question, context, tools_summary, pending_tool = inputs
        
        # Build action space dynamically
        action_space = """
### ACTION SPACE

[1] search_tools
  Description: Search for existing tools that might help solve the problem
  Parameters:
    - query (str): Search query to find relevant tools

[2] create_tool
  Description: Create a new reusable tool if no existing tool fits the need
  Parameters:
    - tool_name (str): Name for the new tool (valid Python function name)
    - description (str): What the tool does
    - reason (str): Why this tool would be useful for future problems

[3] use_tool
  Description: Execute an existing tool with specific inputs
  Parameters:
    - tool_name (str): Name of the tool to use
    - inputs (dict): Input parameters for the tool

[4] answer
  Description: Provide the final answer based on current knowledge
  Parameters:
    - answer (str): The final answer to the question
"""
        
        prompt = f"""
### CONTEXT
You are a self-evolving agent that can create and use tools.
Question: {question}
Previous Actions: {context}

### AVAILABLE TOOLS
{tools_summary}

{action_space}

### INSTRUCTIONS
1. If you need a capability you don't have, first search_tools to see if it exists
2. If no suitable tool exists and the capability would be reusable, use create_tool
3. If a tool exists that can help, use use_tool with the right inputs
4. If you have enough information, use answer to provide the final response
5. Prefer using existing tools over creating new ones
6. Only create tools for genuinely reusable operations

### NEXT ACTION
Decide the next action. Return YAML:

```yaml
thinking: |
    <your reasoning about what to do next>
action: search_tools OR create_tool OR use_tool OR answer
reason: <why you chose this action>
# Include parameters based on action:
query: <if search_tools>
tool_name: <if create_tool or use_tool>
description: <if create_tool>
inputs: <if use_tool, as dict>
answer: <if answer>
```
"""
        
        response = call_llm(prompt)
        decision = _safe_yaml_parse(response, {"action": "answer", "answer": "Unable to process", "reason": "Parse failed"})
        
        # Validate action
        valid_actions = {"search_tools", "create_tool", "use_tool", "answer"}
        if decision.get("action") not in valid_actions:
            decision["action"] = "answer"
            decision["answer"] = f"Invalid action selected. Original response: {response[:200]}"
        
        return decision
    
    def post(self, shared: Dict[str, Any], prep_res: Any, exec_res: Dict[str, Any]) -> str:
        action = exec_res.get("action", "answer")
        
        # Store decision details in shared state
        shared["last_decision"] = exec_res
        
        if action == "search_tools":
            shared["search_query"] = exec_res.get("query", shared.get("question", ""))
            print(f"🔍 Searching for tools: {shared['search_query']}")
            
        elif action == "create_tool":
            shared["new_tool_name"] = exec_res.get("tool_name", "unnamed_tool")
            shared["new_tool_description"] = exec_res.get("description", "")
            print(f"🛠️  Planning to create tool: {shared['new_tool_name']}")
            
        elif action == "use_tool":
            shared["execute_tool_name"] = exec_res.get("tool_name", "")
            shared["execute_tool_inputs"] = exec_res.get("inputs", {})
            print(f"⚡ Using tool: {shared['execute_tool_name']}")
            
        elif action == "answer":
            shared["final_answer"] = exec_res.get("answer", "No answer provided")
            print(f"💡 Ready to answer")
        
        return action


class SearchTools(Node):
    """Search the tool registry for relevant tools."""
    
    def prep(self, shared: Dict[str, Any]) -> Tuple[str, ToolRegistry]:
        query = shared.get("search_query", shared.get("question", ""))
        registry = _get_tool_registry(shared)
        return query, registry
    
    def exec(self, inputs: Tuple[str, ToolRegistry]) -> List[Tuple[str, str, float]]:
        query, registry = inputs
        
        results = registry.search(query, top_k=5, min_similarity=0.1)
        
        tool_summaries = []
        for tool, score in results:
            summary = registry.format_tool_for_prompt(tool.metadata.name)
            tool_summaries.append((tool.metadata.name, summary, score))
        
        return tool_summaries
    
    def post(self, shared: Dict[str, Any], prep_res: Any, exec_res: List[Tuple[str, str, float]]) -> str:
        if exec_res:
            # Format results for context
            results_text = "\n\n".join([
                f"Tool: {name} (relevance: {score:.2f})\n{summary}"
                for name, summary, score in exec_res
            ])
            
            prev_context = shared.get("context", "")
            shared["context"] = f"{prev_context}\n\nTOOL SEARCH RESULTS:\n{results_text}"
            shared["found_tools"] = [name for name, _, _ in exec_res]
            
            print(f"📚 Found {len(exec_res)} relevant tools")
        else:
            shared["context"] = shared.get("context", "") + "\n\nTOOL SEARCH: No relevant tools found"
            shared["found_tools"] = []
            print("📭 No relevant tools found")
        
        return "decide"


class CreateTool(Node):
    """Create and register a new tool based on the agent's decision."""
    
    def __init__(self, max_retries: int = 3, wait: float = 1.0):
        super().__init__(max_retries=max_retries, wait=wait)
    
    def prep(self, shared: Dict[str, Any]) -> Tuple[str, str, str, ToolRegistry]:
        tool_name = shared.get("new_tool_name", "new_tool")
        description = shared.get("new_tool_description", "")
        question = shared.get("question", "")
        registry = _get_tool_registry(shared)
        
        return tool_name, description, question, registry
    
    def exec(self, inputs: Tuple[str, str, str, ToolRegistry]) -> Dict[str, Any]:
        tool_name, description, question, registry = inputs
        
        # Clean tool name
        tool_name = re.sub(r'[^a-zA-Z0-9_]', '_', tool_name)
        if tool_name[0].isdigit():
            tool_name = 'tool_' + tool_name
        
        prompt = f"""
Create a reusable Python tool function.

TOOL NAME: {tool_name}
DESCRIPTION: {description}
CONTEXT: This tool should help with questions like: {question}

REQUIREMENTS:
1. Function must be named exactly '{tool_name}'
2. Include type hints for parameters and return value
3. Include a docstring explaining usage
4. Handle edge cases and errors gracefully
5. Only use these allowed imports: math, re, json, datetime, collections, itertools, functools, operator, string, hashlib, base64, urllib.parse, statistics
6. Do NOT use: exec, eval, open, os.system, subprocess, __import__
7. Keep the function focused and reusable

Return YAML with:

```yaml
source_code: |
    def {tool_name}(param1: type, param2: type = default) -> return_type:
        \"\"\"Description of the tool.
        
        Args:
            param1: Description
            param2: Description
            
        Returns:
            Description of return value
        \"\"\"
        # Implementation
        return result

parameters:
    param1:
        type: str
        description: What this parameter is for
        required: true
    param2:
        type: int
        description: Optional parameter
        required: false
        default: 10

return_type: str
return_description: What the function returns

tags:
    - tag1
    - tag2

examples:
    - input:
          param1: "test"
          param2: 5
      output: "expected result"
```
"""
        
        response = call_llm(prompt)
        result = _safe_yaml_parse(response, {})
        
        if "source_code" not in result:
            raise ValueError(f"LLM did not provide source_code. Response: {response[:500]}")
        
        return {
            "tool_name": tool_name,
            "source_code": result["source_code"],
            "description": description or result.get("description", f"Tool: {tool_name}"),
            "parameters": result.get("parameters", {}),
            "return_type": result.get("return_type", "Any"),
            "return_description": result.get("return_description", ""),
            "tags": result.get("tags", []),
            "examples": result.get("examples", [])
        }
    
    def exec_fallback(self, prep_res: Any, exc: Exception) -> Dict[str, Any]:
        tool_name, description, _, _ = prep_res
        return {
            "tool_name": tool_name,
            "error": str(exc),
            "source_code": None
        }
    
    def post(self, shared: Dict[str, Any], prep_res: Any, exec_res: Dict[str, Any]) -> str:
        tool_name, _, _, registry = prep_res
        
        if exec_res.get("error") or not exec_res.get("source_code"):
            error_msg = exec_res.get("error", "No source code generated")
            shared["context"] = shared.get("context", "") + f"\n\nTOOL CREATION FAILED: {error_msg}"
            print(f"❌ Failed to create tool: {error_msg}")
            return "decide"
        
        # Try to register the tool
        try:
            tool = registry.register(
                name=exec_res["tool_name"],
                description=exec_res["description"],
                source_code=exec_res["source_code"],
                parameters=exec_res["parameters"],
                return_type=exec_res["return_type"],
                return_description=exec_res["return_description"],
                tags=exec_res["tags"],
                examples=exec_res["examples"],
                validate=True,
                test_examples=bool(exec_res["examples"])
            )
            
            shared["context"] = shared.get("context", "") + f"\n\nTOOL CREATED: {exec_res['tool_name']} - {exec_res['description']}"
            shared["last_created_tool"] = exec_res["tool_name"]
            
            print(f"✅ Created and registered tool: {exec_res['tool_name']} (v{tool.metadata.version})")
            
        except ToolValidationError as e:
            shared["context"] = shared.get("context", "") + f"\n\nTOOL VALIDATION FAILED: {e}"
            print(f"⚠️  Tool validation failed: {e}")
            
        except Exception as e:
            shared["context"] = shared.get("context", "") + f"\n\nTOOL REGISTRATION ERROR: {e}"
            print(f"❌ Tool registration error: {e}")
        
        return "decide"


class UseTool(Node):
    """Execute a registered tool with given inputs."""
    
    def __init__(self, max_retries: int = 2, wait: float = 0.5):
        super().__init__(max_retries=max_retries, wait=wait)
    
    def prep(self, shared: Dict[str, Any]) -> Tuple[str, Dict[str, Any], ToolRegistry]:
        tool_name = shared.get("execute_tool_name", "")
        inputs = shared.get("execute_tool_inputs", {})
        registry = _get_tool_registry(shared)
        
        # Ensure inputs is a dict
        if not isinstance(inputs, dict):
            try:
                inputs = dict(inputs) if inputs else {}
            except (TypeError, ValueError):
                inputs = {}
        
        return tool_name, inputs, registry
    
    def exec(self, inputs: Tuple[str, Dict[str, Any], ToolRegistry]) -> Dict[str, Any]:
        tool_name, tool_inputs, registry = inputs
        
        if not tool_name:
            return {"error": "No tool name specified", "result": None}
        
        try:
            result = registry.execute(tool_name, tool_inputs)
            return {"result": result, "error": None, "tool_name": tool_name}
            
        except ToolNotFoundError as e:
            return {"error": f"Tool not found: {e}", "result": None, "tool_name": tool_name}
            
        except ToolExecutionError as e:
            return {"error": f"Execution failed: {e}", "result": None, "tool_name": tool_name}
            
        except Exception as e:
            return {"error": f"Unexpected error: {type(e).__name__}: {e}", "result": None, "tool_name": tool_name}
    
    def exec_fallback(self, prep_res: Any, exc: Exception) -> Dict[str, Any]:
        tool_name, _, _ = prep_res
        return {"error": str(exc), "result": None, "tool_name": tool_name}
    
    def post(self, shared: Dict[str, Any], prep_res: Any, exec_res: Dict[str, Any]) -> str:
        tool_name = exec_res.get("tool_name", "unknown")
        
        if exec_res.get("error"):
            shared["context"] = shared.get("context", "") + f"\n\nTOOL EXECUTION FAILED ({tool_name}): {exec_res['error']}"
            print(f"❌ Tool execution failed: {exec_res['error']}")
        else:
            result_str = json.dumps(exec_res["result"], default=str) if exec_res["result"] is not None else "None"
            shared["context"] = shared.get("context", "") + f"\n\nTOOL RESULT ({tool_name}): {result_str}"
            shared["last_tool_result"] = exec_res["result"]
            print(f"✅ Tool executed successfully: {result_str[:200]}")
        
        return "decide"


class AnswerQuestion(Node):
    """Generate final answer based on accumulated context."""
    
    def prep(self, shared: Dict[str, Any]) -> Tuple[str, str, Optional[Any]]:
        question = shared.get("question", "")
        context = shared.get("context", "")
        final_answer = shared.get("final_answer")
        
        return question, context, final_answer
    
    def exec(self, inputs: Tuple[str, str, Optional[Any]]) -> str:
        question, context, final_answer = inputs
        
        # If we already have a final answer, refine it
        if final_answer:
            prompt = f"""
Question: {question}

Research and tool results:
{context}

Draft answer: {final_answer}

Provide a comprehensive, well-formatted final answer based on the above.
If the draft answer is good, you can keep it. Otherwise, improve it.
"""
        else:
            prompt = f"""
Question: {question}

Research and tool results:
{context}

Based on the above context, provide a comprehensive answer to the question.
"""
        
        answer = call_llm(prompt)
        return answer
    
    def post(self, shared: Dict[str, Any], prep_res: Any, exec_res: str) -> str:
        shared["answer"] = exec_res
        print(f"✅ Answer generated")
        return "done"


class ShouldCreateTool(Node):
    """
    Analyze if the current problem-solving pattern should be saved as a reusable tool.
    This enables the agent to learn from experience.
    """
    
    def prep(self, shared: Dict[str, Any]) -> Tuple[str, str, str, ToolRegistry]:
        question = shared.get("question", "")
        context = shared.get("context", "")
        answer = shared.get("answer", "")
        registry = _get_tool_registry(shared)
        
        return question, context, answer, registry
    
    def exec(self, inputs: Tuple[str, str, str, ToolRegistry]) -> Dict[str, Any]:
        question, context, answer, registry = inputs
        
        prompt = f"""
Analyze this problem-solving session to determine if any reusable tool should be created.

QUESTION: {question}

ACTIONS TAKEN:
{context}

FINAL ANSWER: {answer}

CRITERIA FOR CREATING A TOOL:
1. The operation is likely to be useful for future, similar problems
2. It's a self-contained, pure function (no side effects on external systems)
3. It's not already available in the existing tools
4. It's more than just a simple calculation or string operation

Respond with YAML:

```yaml
should_create_tool: true OR false
reason: <why or why not>
# If should_create_tool is true:
tool_name: <suggested name>
tool_description: <what it does>
abstraction_notes: |
    <how to generalize from this specific case to a reusable tool>
```
"""
        
        response = call_llm(prompt)
        result = _safe_yaml_parse(response, {"should_create_tool": False, "reason": "Parse failed"})
        
        return result
    
    def post(self, shared: Dict[str, Any], prep_res: Any, exec_res: Dict[str, Any]) -> str:
        if exec_res.get("should_create_tool"):
            shared["new_tool_name"] = exec_res.get("tool_name", "learned_tool")
            shared["new_tool_description"] = exec_res.get("tool_description", "")
            shared["abstraction_notes"] = exec_res.get("abstraction_notes", "")
            print(f"💡 Learning opportunity detected: {exec_res.get('tool_name')}")
            return "create_tool"
        else:
            print(f"📝 No new tool needed: {exec_res.get('reason', 'N/A')}")
            return "done"


class ListTools(Node):
    """List all registered tools - useful for debugging and exploration."""
    
    def prep(self, shared: Dict[str, Any]) -> ToolRegistry:
        return _get_tool_registry(shared)
    
    def exec(self, registry: ToolRegistry) -> List[Dict[str, Any]]:
        tools = registry.list_tools(enabled_only=True, limit=100)
        
        return [
            {
                "name": t.metadata.name,
                "description": t.metadata.description,
                "version": t.metadata.version,
                "usage_count": t.metadata.usage_count,
                "tags": t.metadata.tags
            }
            for t in tools
        ]
    
    def post(self, shared: Dict[str, Any], prep_res: Any, exec_res: List[Dict[str, Any]]) -> str:
        shared["tools_list"] = exec_res
        
        if exec_res:
            print(f"📋 Registered tools ({len(exec_res)}):")
            for t in exec_res:
                print(f"  - {t['name']} (v{t['version']}, used {t['usage_count']}x): {t['description'][:60]}...")
        else:
            print("📋 No tools registered yet")
        
        return "done"
