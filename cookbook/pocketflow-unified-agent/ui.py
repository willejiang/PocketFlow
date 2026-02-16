"""
Terminal UI for the Unified Agent

Provides interfaces for:
- Selecting which cookbooks to enable
- Selecting model provider
- Running the agent interactively
- Viewing cookbook information
"""

import os
import sys
from typing import List, Dict, Any, Optional, Set
from pathlib import Path

from adapters import (
    CookbookAdapter,
    AdapterRegistry,
    get_adapter_registry,
    discover_cookbooks,
    load_cookbook_adapter,
    CookbookInfo,
)


def clear_screen():
    """Clear terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header(title: str):
    """Print a formatted header."""
    width = 70
    print("=" * width)
    print(f" {title}".center(width))
    print("=" * width)


def print_cookbook_list(
    cookbooks: List[CookbookInfo],
    selected: Set[str],
    start_idx: int = 0,
    page_size: int = 15
) -> int:
    end_idx = min(start_idx + page_size, len(cookbooks))
    
    print(f"\n{'#':<4} {'Status':<8} {'Name':<35} {'Tags':<20}")
    print("-" * 70)
    
    for i, cb in enumerate(cookbooks[start_idx:end_idx], start=start_idx + 1):
        status = "[X]" if cb.name in selected else "[ ]"
        tags = ", ".join(cb.tags[:3]) if cb.tags else ""
        name = cb.title[:33] + ".." if len(cb.title) > 35 else cb.title
        print(f"{i:<4} {status:<8} {name:<35} {tags:<20}")
    
    print("-" * 70)
    print(f"Showing {start_idx + 1}-{end_idx} of {len(cookbooks)} cookbooks")
    
    return end_idx - start_idx


def model_selection_ui() -> Optional[str]:
    """Interactive UI for selecting a model provider."""
    presets = [
        ("OpenAI GPT-4o", "openai:gpt-4o"),
        ("OpenAI GPT-4o Mini", "openai:gpt-4o-mini"),
        ("OpenAI O1 Preview (thinking)", "openai:o1-preview"),
        ("OpenAI O1 Mini (thinking)", "openai:o1-mini"),
        ("Anthropic Claude 3.5 Sonnet", "anthropic:claude-3-5-sonnet-20241022"),
        ("Anthropic Claude 3.7 Sonnet (thinking)", "anthropic:claude-3-7-sonnet-20250219"),
        ("Anthropic Claude 3 Opus", "anthropic:claude-3-opus-20240229"),
        ("Ollama Llama 3.2", "ollama:llama3.2"),
        ("Ollama Llama 3.2 Vision", "ollama:llama3.2-vision"),
        ("Ollama Mistral", "ollama:mistral"),
        ("Ollama Qwen 2.5", "ollama:qwen2.5"),
        ("Ollama DeepSeek R1 (thinking)", "ollama:deepseek-r1"),
        ("Custom Transformers model", "transformers:"),
        ("Custom vLLM model", "vllm:"),
        ("Custom vLLM server", "vllm-server:"),
    ]
    
    while True:
        clear_screen()
        print_header("Model Selection")
        
        print("\nSelect a model provider:\n")
        
        for i, (name, _) in enumerate(presets, 1):
            print(f"  {i:2}. {name}")
        
        print(f"\n  c.  Custom specification")
        print(f"  d.  Use default (auto-detect)")
        print(f"  q.  Cancel")
        
        try:
            choice = input("\n> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return None
        
        if choice == 'q':
            return None
        
        if choice == 'd':
            return None  # Will use auto-detect
        
        if choice == 'c':
            print("\nEnter model specification:")
            print("  Format: <provider>:<model_id>[:api_base]")
            print("  Examples:")
            print("    openai:gpt-4o")
            print("    ollama:llama3.2")
            print("    transformers:meta-llama/Llama-3.2-3B-Instruct")
            print("    vllm-server:http://localhost:8000:model_name")
            
            try:
                spec = input("\n> ").strip()
                if spec:
                    return spec
            except (EOFError, KeyboardInterrupt):
                pass
            continue
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(presets):
                name, spec = presets[idx]
                
                if spec.endswith(":"):
                    print(f"\nEnter model ID for {name.split()[0]}:")
                    if "transformers" in spec:
                        print("  Example: meta-llama/Llama-3.2-3B-Instruct")
                    elif "vllm-server" in spec:
                        print("  Enter: <server_url>:<model_name>")
                        print("  Example: http://localhost:8000:llama3")
                    else:
                        print("  Example: mistralai/Mistral-7B-Instruct-v0.3")
                    
                    try:
                        model_id = input("\n> ").strip()
                        if model_id:
                            return spec + model_id
                    except (EOFError, KeyboardInterrupt):
                        pass
                    continue
                
                return spec
        except ValueError:
            pass
        
        print("\nInvalid choice. Press Enter to continue...")
        input()


def cookbook_selection_ui(cookbooks: List[CookbookInfo]) -> List[str]:
    """Interactive UI for selecting cookbooks."""
    selected: Set[str] = set()
    page = 0
    page_size = 15
    
    common_cookbooks = {
        "pocketflow-agent",
        "pocketflow-rag",
        "pocketflow-thinking",
        "pocketflow-code-generator",
        "pocketflow-tool-search",
    }
    
    for cb in cookbooks:
        if cb.name in common_cookbooks:
            selected.add(cb.name)
    
    while True:
        clear_screen()
        print_header("Unified Agent - Cookbook Selection")
        
        print("\nSelect which cookbooks to enable for this session.")
        print("The agent will be able to use capabilities from selected cookbooks.\n")
        
        start_idx = page * page_size
        print_cookbook_list(cookbooks, selected, start_idx, page_size)
        
        total_pages = (len(cookbooks) + page_size - 1) // page_size
        
        print(f"\nSelected: {len(selected)} cookbooks")
        print("\nCommands:")
        print("  <number>  - Toggle cookbook selection")
        print("  a         - Select all")
        print("  n         - Select none")
        print("  t <tag>   - Toggle all with tag (e.g., 't agent')")
        if page > 0:
            print("  p         - Previous page")
        if page < total_pages - 1:
            print("  x         - Next page")
        print("  v <num>   - View cookbook details")
        print("  m         - Select model provider")
        print("  d         - Done, start agent")
        print("  q         - Quit")
        
        try:
            cmd = input("\n> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            sys.exit(0)
        
        if cmd == 'q':
            print("Exiting...")
            sys.exit(0)
        
        elif cmd == 'd':
            if not selected:
                print("\nNo cookbooks selected. Select at least one or press 'q' to quit.")
                input("Press Enter to continue...")
            else:
                return list(selected)
        
        elif cmd == 'm':
            model_spec = model_selection_ui()
            if model_spec:
                try:
                    from main import setup_model
                    setup_model(model_spec)
                    print(f"\nModel configured: {model_spec}")
                    input("Press Enter to continue...")
                except Exception as e:
                    print(f"\nError setting up model: {e}")
                    input("Press Enter to continue...")
        
        elif cmd == 'a':
            selected = {cb.name for cb in cookbooks}
        
        elif cmd == 'n':
            selected.clear()
        
        elif cmd == 'p' and page > 0:
            page -= 1
        
        elif cmd == 'x' and page < total_pages - 1:
            page += 1
        
        elif cmd.startswith('t '):
            tag = cmd[2:].strip()
            matching = [cb for cb in cookbooks if tag in cb.tags]
            if matching:
                all_selected = all(cb.name in selected for cb in matching)
                for cb in matching:
                    if all_selected:
                        selected.discard(cb.name)
                    else:
                        selected.add(cb.name)
                print(f"\n{'Deselected' if all_selected else 'Selected'} {len(matching)} cookbooks with tag '{tag}'")
            else:
                print(f"\nNo cookbooks found with tag '{tag}'")
            input("Press Enter to continue...")
        
        elif cmd.startswith('v '):
            try:
                num = int(cmd[2:].strip())
                if 1 <= num <= len(cookbooks):
                    show_cookbook_details(cookbooks[num - 1])
                else:
                    print(f"\nInvalid number. Enter 1-{len(cookbooks)}")
                input("\nPress Enter to continue...")
            except ValueError:
                print("\nInvalid number")
                input("Press Enter to continue...")
        
        elif cmd.isdigit():
            num = int(cmd)
            if 1 <= num <= len(cookbooks):
                cb = cookbooks[num - 1]
                if cb.name in selected:
                    selected.remove(cb.name)
                else:
                    selected.add(cb.name)
            else:
                print(f"\nInvalid number. Enter 1-{len(cookbooks)}")
                input("Press Enter to continue...")
        
        else:
            if '-' in cmd:
                try:
                    start, end = cmd.split('-')
                    start, end = int(start), int(end)
                    for i in range(start, end + 1):
                        if 1 <= i <= len(cookbooks):
                            selected.add(cookbooks[i - 1].name)
                except ValueError:
                    pass


def show_cookbook_details(cb: CookbookInfo):
    """Show detailed information about a cookbook."""
    clear_screen()
    print_header(f"Cookbook: {cb.title}")
    
    print(f"\nName:        {cb.name}")
    print(f"Path:        {cb.path}")
    print(f"Description: {cb.description[:200]}..." if len(cb.description) > 200 else f"Description: {cb.description}")
    print(f"Tags:        {', '.join(cb.tags) if cb.tags else 'None'}")
    
    print(f"\nFiles:")
    print(f"  - nodes.py:    {'Yes' if cb.has_nodes else 'No'}")
    print(f"  - flow.py:     {'Yes' if cb.has_flow else 'No'}")
    print(f"  - main.py:     {'Yes' if cb.has_main else 'No'}")
    print(f"  - adapter.py:  {'Yes' if cb.has_adapter else 'No'}")
    print(f"  - manifest:    {'Yes' if cb.has_manifest else 'No'}")
    
    if cb.dependencies:
        print(f"\nDependencies:")
        for dep in cb.dependencies[:10]:
            print(f"  - {dep}")
        if len(cb.dependencies) > 10:
            print(f"  ... and {len(cb.dependencies) - 10} more")


def quick_select_ui(cookbooks: List[CookbookInfo]) -> List[str]:
    """Simpler selection UI for non-interactive use."""
    defaults = [
        "pocketflow-agent",
        "pocketflow-rag",
        "pocketflow-thinking",
        "pocketflow-code-generator",
        "pocketflow-text2sql",
        "pocketflow-tool-search",
        "pocketflow-tool-crawler",
        "pocketflow-self-evolving",
    ]
    
    available = {cb.name for cb in cookbooks}
    return [name for name in defaults if name in available]


def select_by_tags(cookbooks: List[CookbookInfo], tags: List[str]) -> List[str]:
    """Select cookbooks that match any of the given tags."""
    selected = []
    for cb in cookbooks:
        if any(tag in cb.tags for tag in tags):
            selected.append(cb.name)
    return selected


def select_by_names(cookbooks: List[CookbookInfo], names: List[str]) -> List[str]:
    """Select cookbooks by name (supports partial matching)."""
    available = {cb.name for cb in cookbooks}
    selected = []
    
    for name in names:
        if name in available:
            selected.append(name)
        else:
            matches = [cb for cb in cookbooks if name in cb.name]
            selected.extend(cb.name for cb in matches)
    
    return list(set(selected))


def run_interactive_agent(
    registry: AdapterRegistry,
    shared: Optional[Dict[str, Any]] = None
):
    """Run the agent in interactive mode."""
    from flow import run_agent_loop
    
    if shared is None:
        shared = {}
    
    clear_screen()
    print_header("Unified Agent - Interactive Mode")
    
    # Show model info
    try:
        from model_providers import get_model_registry
        model_registry = get_model_registry()
        if model_registry.active_name:
            info = model_registry.get_model_info(model_registry.active_name)
            if info:
                print(f"\nModel: {info.get('provider_type', 'unknown')}:{info.get('model_id', 'unknown')}")
                caps = info.get('capabilities', [])
                if caps:
                    print(f"Capabilities: {', '.join(caps)}")
    except Exception:
        pass
    
    adapters = registry.list_adapters()
    print(f"\nLoaded {len(adapters)} cookbooks:")
    for adapter in adapters:
        actions = [a.name for a in adapter.actions]
        print(f"  - {adapter.name}: {len(actions)} actions")
    
    print("\n" + "-" * 70)
    print("Type your question or command. Special commands:")
    print("  /list     - List all available actions")
    print("  /adapters - List loaded adapters")
    print("  /model    - Show/change model")
    print("  /stats    - Show usage statistics")
    print("  /help     - Show help")
    print("  /quit     - Exit")
    print("-" * 70 + "\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break
        
        if not user_input:
            continue
        
        if user_input.startswith("/"):
            handle_command(user_input, registry, shared)
            continue
        
        print("\nAgent: ", end="", flush=True)
        
        try:
            result = run_agent_loop(user_input, registry, shared)
            print(result)
        except Exception as e:
            print(f"Error: {e}")
        
        print()


def handle_command(cmd: str, registry: AdapterRegistry, shared: Dict[str, Any]):
    """Handle special commands."""
    cmd_lower = cmd.lower()
    
    if cmd_lower in ("/quit", "/exit", "/q"):
        print("\nGoodbye!")
        sys.exit(0)
    
    elif cmd_lower == "/list":
        print("\nAvailable actions:")
        for adapter_name, action in registry.list_all_actions():
            print(f"  {action.name}: {action.description[:50]}...")
    
    elif cmd_lower == "/adapters":
        print("\nLoaded adapters:")
        for adapter in registry.list_adapters(enabled_only=False):
            status = "enabled" if adapter.enabled else "disabled"
            print(f"  {adapter.name} ({status}): {len(adapter.actions)} actions")
    
    elif cmd_lower == "/model" or cmd_lower.startswith("/model "):
        parts = cmd.split(maxsplit=1)
        
        if len(parts) == 1:
            # Show current model
            try:
                from model_providers import get_model_registry
                model_registry = get_model_registry()
                if model_registry.active_name:
                    info = model_registry.get_model_info(model_registry.active_name)
                    print(f"\nCurrent model:")
                    print(f"  Provider: {info.get('provider_type', 'unknown')}")
                    print(f"  Model ID: {info.get('model_id', 'unknown')}")
                    print(f"  Capabilities: {', '.join(info.get('capabilities', []))}")
                else:
                    print("\nNo model configured")
            except Exception as e:
                print(f"\nError: {e}")
        else:
            # Change model
            model_spec = parts[1]
            try:
                from main import setup_model
                setup_model(model_spec)
                print(f"\nModel changed to: {model_spec}")
            except Exception as e:
                print(f"\nError changing model: {e}")
    
    elif cmd_lower == "/stats":
        print("\nUsage statistics:")
        for stat in registry.get_stats():
            print(f"  {stat['name']}:")
            for action_name, action_stats in stat.get('actions', {}).items():
                print(f"    {action_name}: {action_stats['calls']} calls, "
                      f"{action_stats['successes']} successes")
    
    elif cmd_lower == "/help":
        print("\nHelp:")
        print("  Type any question to get an answer from the agent.")
        print("  The agent will use available cookbook capabilities.")
        print("\n  Commands:")
        print("    /list     - List all available actions")
        print("    /adapters - List loaded adapters")
        print("    /model    - Show current model")
        print("    /model <spec> - Change model (e.g., /model ollama:llama3.2)")
        print("    /stats    - Show usage statistics")
        print("    /help     - Show this help")
        print("    /quit     - Exit")
    
    else:
        print(f"\nUnknown command: {cmd}")
        print("Type /help for available commands.")


def print_action_list(registry: AdapterRegistry):
    """Print all available actions grouped by adapter."""
    print("\nAvailable Capabilities:")
    print("=" * 70)
    
    for adapter in registry.list_adapters():
        print(f"\n{adapter.name}")
        print(f"  {adapter.description}")
        print("  Actions:")
        for action in adapter.actions:
            print(f"    - {action.name}: {action.description[:50]}...")
    
    print("\n" + "=" * 70)
