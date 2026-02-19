"""
Unified Agent Flow

The main flow that connects decision and execution nodes.
"""

from typing import Dict, Any, List, Optional

from pocketflow import Flow

from nodes import DecideAction, ExecuteAction, FinalizeAnswer
from adapters import (
    AdapterRegistry,
    get_adapter_registry,
    discover_cookbooks,
    load_cookbook_adapter,
    get_system_tools_adapter,
)


def create_unified_flow(max_iterations: int = 100) -> Flow:
    """
    Create the unified agent flow.
    
    Flow structure:
        [DecideAction] ◄─────┐
              │              │
              ▼              │
        [ExecuteAction] ─────┘
              │
         (if done)
              │
              ▼
        [FinalizeAnswer]
    """
    decide = DecideAction(max_iterations=max_iterations)
    execute = ExecuteAction()
    finalize = FinalizeAnswer()
    
    decide - "execute" >> execute
    execute - "decide" >> decide
    execute - "done" >> finalize
    
    return Flow(start=decide)


def load_adapters(
    cookbook_names: List[str],
    registry: Optional[AdapterRegistry] = None,
    include_system_tools: bool = True,
    allow_commands: bool = True
) -> AdapterRegistry:
    """
    Load adapters for the specified cookbooks.
    
    Args:
        cookbook_names: List of cookbook names to load
        registry: Optional existing registry to use
        include_system_tools: Whether to include built-in system tools (file/command ops)
        allow_commands: Whether to allow command execution in system tools
        
    Returns:
        AdapterRegistry with loaded adapters
    """
    if registry is None:
        AdapterRegistry.reset()
        registry = get_adapter_registry()
    
    # Always load system tools first (file operations, commands, etc.)
    if include_system_tools:
        try:
            system_adapter = get_system_tools_adapter(allow_commands=allow_commands)
            registry.register(system_adapter)
            print("Loaded system tools (file operations, command execution)")
        except Exception as e:
            print(f"Warning: Failed to load system tools: {e}")
    
    # Discover all cookbooks
    all_cookbooks = discover_cookbooks()
    cookbook_map = {cb.name: cb for cb in all_cookbooks}
    
    # Load requested adapters
    loaded = 0
    failed = []
    
    for name in cookbook_names:
        if name not in cookbook_map:
            print(f"Warning: Cookbook '{name}' not found")
            failed.append(name)
            continue
        
        cb_info = cookbook_map[name]
        
        try:
            adapter = load_cookbook_adapter(cb_info, auto_generate=True)
            if adapter:
                registry.register(adapter)
                loaded += 1
            else:
                failed.append(name)
        except Exception as e:
            print(f"Warning: Failed to load adapter for '{name}': {e}")
            failed.append(name)
    
    print(f"Loaded {loaded}/{len(cookbook_names)} cookbook adapters")
    if failed:
        print(f"Failed to load: {', '.join(failed)}")
    
    return registry


def run_agent(
    question: str,
    cookbook_names: List[str],
    max_iterations: int = 100,
    shared: Optional[Dict[str, Any]] = None,
    include_system_tools: bool = True,
    allow_commands: bool = True
) -> str:
    """
    Run the unified agent with specified cookbooks.
    
    Args:
        question: The question to answer
        cookbook_names: List of cookbook names to enable
        max_iterations: Maximum decision iterations
        shared: Optional initial shared state
        include_system_tools: Whether to include built-in system tools
        allow_commands: Whether to allow command execution
        
    Returns:
        The final answer
    """
    # Setup shared state
    if shared is None:
        shared = {}
    
    shared["question"] = question
    
    # Load adapters
    registry = load_adapters(
        cookbook_names,
        include_system_tools=include_system_tools,
        allow_commands=allow_commands
    )
    shared["adapter_registry"] = registry
    
    # Initialize adapters
    registry.initialize_all(shared)
    
    # Create and run flow
    flow = create_unified_flow(max_iterations=max_iterations)
    
    try:
        flow.run(shared)
    finally:
        registry.cleanup_all(shared)
    
    return shared.get("final_answer", shared.get("answer", "No answer generated"))


def run_agent_loop(
    question: str,
    registry: AdapterRegistry,
    shared: Dict[str, Any]
) -> str:
    """
    Run a single agent loop with an existing registry.
    Used for interactive mode.
    """
    shared["question"] = question
    shared["context"] = shared.get("context", "")
    shared["iteration"] = 0
    shared["adapter_registry"] = registry  # Critical: pass registry to nodes
    
    # Initialize adapters if not already done
    registry.initialize_all(shared)
    
    flow = create_unified_flow()
    flow.run(shared)
    
    return shared.get("final_answer", shared.get("answer", "No answer"))
