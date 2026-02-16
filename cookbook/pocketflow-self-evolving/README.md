# Self-Evolving Agent

An agent that can create, register, search, and use tools dynamically. The agent expands its action pool over time by learning from problem-solving patterns.

## Overview

This implementation adds "self-evolving" capabilities to PocketFlow agents:

1. **Tool Registry**: Persistent storage for dynamically created tools with metadata, versioning, and usage tracking
2. **Tool Creation**: Agent can create new reusable tools when it recognizes a pattern
3. **Tool Search**: Semantic or keyword-based search to find relevant existing tools
4. **Tool Execution**: Safe sandboxed execution of registered tools
5. **Learning**: After solving a problem, the agent analyzes if a reusable tool should be created

## Architecture

```
                    ┌─────────────────────────────────────────────────┐
                    │                                                 │
                    ▼                                                 │
                [DecideAction] ─── search_tools ──► [SearchTools] ───┘
                    │                                                 
                    ├── create_tool ──► [CreateTool] ────────────────┘
                    │                                                 
                    ├── use_tool ──► [UseTool] ──────────────────────┘
                    │                                                 
                    └── answer ──► [AnswerQuestion] ──► [ShouldCreateTool]
                                                            │
                                                create_tool │
                                                            ▼
                                                      [CreateTool]
```

## Key Components

### Tool Registry (`tools/tool_registry.py`)

Thread-safe, persistent storage for tools using SQLite:

```python
from tools.tool_registry import ToolRegistry, get_registry

# Get or create registry
registry = get_registry("my_tools.db")

# Register a tool
registry.register(
    name="calculate_gcd",
    description="Calculate the greatest common divisor of two numbers",
    source_code='''
def calculate_gcd(a: int, b: int) -> int:
    """Calculate GCD using Euclidean algorithm."""
    while b:
        a, b = b, a % b
    return a
''',
    parameters={
        "a": {"type": "int", "description": "First number", "required": True},
        "b": {"type": "int", "description": "Second number", "required": True}
    },
    return_type="int",
    return_description="The GCD of a and b",
    tags=["math", "arithmetic"],
    examples=[
        {"input": {"a": 48, "b": 18}, "output": 6}
    ]
)

# Search for tools
results = registry.search("calculate greatest common divisor")

# Execute a tool
result = registry.execute("calculate_gcd", {"a": 48, "b": 18})
print(result)  # 6

# Get tool statistics
stats = registry.get_stats("calculate_gcd")
```

### Agent Nodes (`nodes.py`)

- **DecideAction**: Chooses between search_tools, create_tool, use_tool, or answer
- **SearchTools**: Searches the registry for relevant tools
- **CreateTool**: Creates and registers a new tool via LLM
- **UseTool**: Executes a registered tool
- **AnswerQuestion**: Provides the final answer
- **ShouldCreateTool**: Analyzes if a solved problem should become a tool

### Flow (`flow.py`)

```python
from flow import create_self_evolving_agent_flow

# Create the agent flow
agent = create_self_evolving_agent_flow(enable_learning=True)

# Run with a question
shared = {"question": "What is the GCD of 48 and 18?"}
agent.run(shared)

print(shared["answer"])
```

## Usage

### Basic Usage

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key"

# Run with a question
python main.py --"What is the factorial of 10?"

# List registered tools
python main.py --list-tools

# Run the demo
python main.py --demo
```

### Programmatic Usage

```python
from flow import create_self_evolving_agent_flow

agent = create_self_evolving_agent_flow()
shared = {
    "question": "Convert 100 Fahrenheit to Celsius",
    "tool_registry_path": "my_tools.db"  # Optional: custom database
}

agent.run(shared)
print(shared["answer"])
```

## Tool Security

The registry validates tools before registration:

- **Syntax validation**: Code must compile
- **Dangerous pattern detection**: Blocks eval, exec, open, subprocess, etc.
- **Import restrictions**: Only allows safe imports (math, re, json, etc.)
- **Example testing**: If examples provided, they must pass

## Features

### Versioning

Tools are versioned automatically. Each update creates a new version while preserving history:

```python
registry.register(name="my_tool", ...)  # version 1
registry.register(name="my_tool", ...)  # version 2

# Get specific version
old_tool = registry.get("my_tool", version=1)
```

### Usage Tracking

The registry tracks execution statistics:

```python
stats = registry.get_stats("my_tool")
# {
#   "total_executions": 42,
#   "successful_executions": 40,
#   "failure_rate": 0.048,
#   "avg_execution_time_ms": 15.3,
#   ...
# }
```

### Enable/Disable Tools

```python
registry.disable("problematic_tool")  # Stop using it
registry.enable("problematic_tool")   # Re-enable it
```

## Testing

```bash
# Run all tests
python -m pytest test_tool_registry.py test_nodes.py -v

# Run only registry tests
python -m pytest test_tool_registry.py -v

# Run only node tests
python -m pytest test_nodes.py -v
```

## Example Session

```
$ python main.py --"What is the GCD of 48 and 18?"
============================================================
🤖 Self-Evolving Agent
============================================================
Question: What is the GCD of 48 and 18?
------------------------------------------------------------
🔍 Searching for tools: gcd calculator math
📭 No relevant tools found
🛠️  Planning to create tool: calculate_gcd
✅ Created and registered tool: calculate_gcd (v1)
⚡ Using tool: calculate_gcd
✅ Tool executed successfully: 6
💡 Ready to answer
✅ Answer generated
------------------------------------------------------------
📝 Final Answer:
The greatest common divisor (GCD) of 48 and 18 is 6.
============================================================
```

On subsequent runs with similar questions:

```
$ python main.py --"What is the GCD of 100 and 35?"
...
🔍 Searching for tools: gcd calculator
📚 Found 1 relevant tools
⚡ Using tool: calculate_gcd
✅ Tool executed successfully: 5
...
```

## Requirements

```
openai>=1.0.0
pyyaml>=6.0
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key (required) | - |
| `TOOL_REGISTRY_DB` | Path to tool database | `tool_registry.db` |
