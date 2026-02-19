# Unified Agent

A unified agent that can dynamically load and use capabilities from **any** PocketFlow cookbook. Instead of isolated examples, cookbooks are treated as pluggable modules that can be selected and combined at runtime.

## Key Features

- **Dynamic Cookbook Loading**: Automatically discovers and loads cookbooks at runtime
- **Pluggable Adapters**: Each cookbook can provide an adapter to expose its capabilities
- **Auto-Generation**: Cookbooks without explicit adapters get auto-generated ones
- **Selection UI**: Interactive terminal UI to select which cookbooks to enable
- **Tag-Based Selection**: Select cookbooks by tags (e.g., all "agent" or "search" cookbooks)
- **Extensible**: Easy to add new cookbooks without modifying the unified agent

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Unified Agent                                │
├─────────────────────────────────────────────────────────────────────┤
│                       AdapterRegistry                                │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │  agent   │ │   rag    │ │ thinking │ │   code   │ │   ...    │  │
│  │ adapter  │ │ adapter  │ │ adapter  │ │ adapter  │ │          │  │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘  │
│       │            │            │            │            │         │
│       ▼            ▼            ▼            ▼            ▼         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │pocketflow│ │pocketflow│ │pocketflow│ │pocketflow│ │ Other    │  │
│  │  -agent  │ │  -rag    │ │-thinking │ │  -code   │ │ cookbook │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
                         ┌───────────────┐
                         │ DecideAction  │ ← LLM chooses from ALL loaded actions
                         └───────┬───────┘
                                 │
                                 ▼
                         ┌───────────────┐
                         │ ExecuteAction │ ← Registry routes to correct adapter
                         └───────────────┘
```

## Usage

### Interactive Mode (Cookbook Selection UI)

```bash
python main.py
```

This opens an interactive UI where you can:
- Browse all available cookbooks
- Toggle selection with numbers
- Filter by tags
- View cookbook details

### Quick Run with Defaults

```bash
python main.py --quick "What is the capital of France?"
```

Uses a default set of commonly useful cookbooks.

### Select Specific Cookbooks

```bash
python main.py --select agent,rag,thinking "Your question here"
```

### Select by Tags

```bash
python main.py --tags search,agent "Find information about AI"
```

### List Available Cookbooks

```bash
python main.py --list
```

### Show Available Actions

```bash
python main.py --actions
```

## How Adapters Work

Each cookbook can expose its capabilities through an **adapter**. There are three ways to create an adapter:

### 1. Explicit adapter.py (Recommended)

Create an `adapter.py` file in your cookbook directory:

```python
from adapters.base import CookbookAdapter, AdapterAction

class MyAdapter(CookbookAdapter):
    @property
    def name(self):
        return "pocketflow-my-cookbook"
    
    @property
    def description(self):
        return "Description of my cookbook"
    
    @property
    def actions(self):
        return [
            AdapterAction(
                name="my_action",
                description="What this action does",
                parameters={
                    "input": {"type": "str", "description": "Input", "required": True}
                }
            )
        ]
    
    def execute(self, action_name, params, shared):
        if action_name == "my_action":
            # Your logic here
            return {"success": True, "result": "Done"}
        return {"success": False, "error": "Unknown action"}

def get_adapter():
    return MyAdapter()
```

### 2. cookbook_manifest.yaml

Create a YAML manifest describing your cookbook:

```yaml
name: pocketflow-my-cookbook
description: What my cookbook does
tags: [agent, tool]

actions:
  - name: my_action
    description: What this action does
    parameters:
      input:
        type: str
        description: Input parameter
        required: true
```

### 3. Auto-Generation

If neither adapter.py nor manifest exists, the unified agent will analyze the cookbook's code (nodes.py, flow.py) and auto-generate an adapter.

## Adding a New Cookbook

1. Create your cookbook in the `cookbook/` directory as `pocketflow-your-name/`
2. Implement your nodes and flow as usual
3. (Optional) Create `adapter.py` for explicit control
4. (Optional) Create `cookbook_manifest.yaml` for declarative configuration
5. The unified agent will automatically discover your cookbook

## Built-in System Tools

The unified agent includes built-in system tools for fundamental operations that are always available:

### File Operations
| Action | Description |
|--------|-------------|
| `read_file` | Read file contents (with optional line range) |
| `write_file` | Write/append content to a file |
| `list_directory` | List files and directories (with glob patterns) |
| `file_exists` | Check if a file or directory exists |
| `delete_file` | Delete a file |
| `create_directory` | Create directories (including parents) |

### Command Execution
| Action | Description |
|--------|-------------|
| `run_command` | Execute shell commands with timeout |

### Why Built-in?

These system tools cannot be created by the self-evolving agent due to security restrictions. The tool registry explicitly blocks dangerous operations like `open()`, `subprocess`, `os.system` to ensure dynamically created tools are safe. The built-in system tools provide these fundamental capabilities with proper security controls:

- File operations are restricted to a configurable working directory
- Command execution has configurable timeout
- Output is truncated to prevent memory issues
- Commands can be disabled entirely if needed

### Disabling System Tools

```python
# Run without system tools
from flow import run_agent
answer = run_agent(
    question="Your question",
    cookbook_names=["pocketflow-agent"],
    include_system_tools=False  # Disable file/command ops
)

# Run with system tools but no command execution
answer = run_agent(
    question="Your question",
    cookbook_names=["pocketflow-agent"],
    allow_commands=False  # File ops only
)
```

## Available Cookbooks

Run `python main.py --list` to see all available cookbooks grouped by tags:

- **agent**: Research agents, web search
- **rag**: Document indexing and retrieval
- **thinking**: Chain of thought, planning
- **code**: Code generation and execution
- **sql**: Database queries
- **search**: Web search, crawling
- **chat**: Conversation, memory
- **workflow**: Multi-step workflows
- And more...

## Programmatic Usage

```python
from flow import run_agent, load_adapters
from adapters import get_adapter_registry

# Simple usage
answer = run_agent(
    question="What is machine learning?",
    cookbook_names=["pocketflow-agent", "pocketflow-rag"]
)

# More control
registry = load_adapters(["pocketflow-agent", "pocketflow-thinking"])
shared = {"question": "Complex problem..."}

from flow import create_unified_flow
flow = create_unified_flow(max_iterations=10)
flow.run(shared)

print(shared["final_answer"])
```

## Templates

See the `templates/` directory for:
- `adapter_template.py` - Template for creating explicit adapters
- `cookbook_manifest.yaml` - Template for manifest-based configuration

## Testing

```bash
python -m pytest test_adapters.py -v
```

## Project Structure

```
pocketflow-unified-agent/
├── adapters/
│   ├── __init__.py
│   ├── base.py          # CookbookAdapter, AdapterAction, AdapterRegistry
│   ├── discovery.py     # Cookbook discovery and auto-generation
│   └── system_tools.py  # Built-in file/command operations
├── model_providers/
│   ├── __init__.py
│   ├── base.py          # ModelProvider base class
│   ├── factory.py       # Provider factory
│   ├── openai_provider.py
│   ├── anthropic_provider.py
│   ├── ollama_provider.py
│   ├── transformers_provider.py
│   └── vllm_provider.py
├── templates/
│   ├── adapter_template.py
│   └── cookbook_manifest.yaml
├── utils/
│   └── call_llm.py
├── nodes.py             # DecideAction, ExecuteAction nodes
├── flow.py              # Unified flow
├── ui.py                # Selection UI
├── main.py              # CLI entry point
├── test_adapters.py
└── README.md
```

## Comparison: Before vs After

### Before (Isolated Cookbooks)
```
pocketflow-agent/     → Only web search
pocketflow-rag/       → Only RAG
pocketflow-thinking/  → Only chain of thought

# Cannot combine capabilities!
```

### After (Unified Agent)
```python
# Select any combination of cookbooks
python main.py --select agent,rag,thinking,code "Complex question"

# The agent can now:
# - Search the web (from agent)
# - Retrieve from indexed documents (from rag)
# - Think step by step (from thinking)
# - Generate and run code (from code)
# All in one session!
```

## Model Selection

The unified agent supports multiple model providers. You can select the model at runtime via CLI or interactive UI.

### CLI Model Selection

```bash
# OpenAI (default)
python main.py --model openai:gpt-4o "Your question"
python main.py --model openai:o1-preview "Complex reasoning task"  # thinking model

# Anthropic
python main.py --model anthropic:claude-3-5-sonnet-20241022 "Your question"
python main.py --model anthropic:claude-3-7-sonnet-20250219 "Reasoning task"  # thinking

# Ollama (local)
python main.py --model ollama:llama3.2 "Your question"
python main.py --model ollama:llama3.2-vision "Describe this image"  # vision

# Transformers (local HuggingFace)
python main.py --model transformers:meta-llama/Llama-3.2-3B-Instruct "Your question"
python main.py --model transformers:Qwen/Qwen2-VL-7B-Instruct "Vision task"  # vision

# vLLM (optimized local inference)
python main.py --model vllm:mistralai/Mistral-7B-Instruct-v0.3 "Your question"

# vLLM Server (remote)
python main.py --model vllm-server:http://localhost:8000:model_name "Your question"
```

### List Available Models

```bash
python main.py --models
```

### Interactive Model Selection

In interactive mode, press `m` to change the model:
```
> /model                              # Show current model
> /model ollama:llama3.2              # Change to Ollama
> /model anthropic:claude-3-5-sonnet  # Change to Claude
```

### Model Capabilities

| Capability | Description | Examples |
|------------|-------------|----------|
| TEXT | Standard text generation | All models |
| VISION | Image understanding | GPT-4o, Claude 3.x, LLaVA, Qwen-VL |
| THINKING | Extended reasoning | O1, Claude 3.7, DeepSeek-R1, QwQ |
| STREAMING | Stream responses | Most API models |

### Programmatic Model Selection

```python
from model_providers import ModelConfig, create_provider, get_model_registry

# Create a specific model config
config = ModelConfig.anthropic(model_id="claude-3-5-sonnet-20241022")
provider = create_provider(config)
provider.initialize()

# Register as the active model
registry = get_model_registry()
registry.register_provider("claude", provider)
registry.set_active("claude")

# Now the unified agent will use this model
from flow import run_agent
answer = run_agent("Your question", ["pocketflow-agent"])
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | Required for OpenAI models |
| `ANTHROPIC_API_KEY` | Required for Anthropic models |
| `POCKETFLOW_COOKBOOK` | Optional: Path to cookbook directory |
