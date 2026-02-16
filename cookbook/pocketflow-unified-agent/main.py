#!/usr/bin/env python3
"""
Unified Agent - Main Entry Point

A unified agent that can dynamically load and use capabilities from ANY cookbook.

Usage:
    python main.py                      # Interactive cookbook selection UI
    python main.py --interactive        # Select cookbooks then run interactively
    python main.py --list               # List all available cookbooks
    python main.py --quick "question"   # Quick run with default cookbooks
    python main.py --select agent,rag "question"  # Run with specific cookbooks
    
Model Selection:
    python main.py --model openai:gpt-4o "question"
    python main.py --model ollama:llama3.2 "question"
    python main.py --model transformers:meta-llama/Llama-3.2-3B-Instruct "question"
    python main.py --model vllm:mistralai/Mistral-7B-Instruct-v0.3 "question"
    python main.py --model vllm-server:http://localhost:8000:model_name "question"
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from adapters import (
    discover_cookbooks,
    load_cookbook_adapter,
    get_adapter_registry,
    AdapterRegistry,
)
from flow import load_adapters, run_agent, create_unified_flow
from ui import (
    cookbook_selection_ui,
    quick_select_ui,
    select_by_tags,
    select_by_names,
    run_interactive_agent,
    print_action_list,
    print_header,
    clear_screen,
)


_model_provider = None


def setup_model(model_spec: str = None, api_key: str = None):
    """Setup the model provider based on specification."""
    global _model_provider
    
    from model_providers import (
        get_model_registry,
        create_from_string,
        setup_default_model,
        ModelConfig,
        create_provider,
    )
    from utils.call_llm import set_provider
    
    registry = get_model_registry()
    
    if model_spec:
        provider = create_from_string(model_spec)
        if api_key:
            provider.config.api_key = api_key
        provider.initialize()
        registry.register_provider("active", provider)
        registry.set_active("active")
        _model_provider = provider
        set_provider(provider)
        
        print(f"Using model: {provider.config.provider_type}:{provider.config.model_id}")
        caps = [c.name for c in provider.capabilities]
        print(f"Capabilities: {', '.join(caps)}")
        return provider
    
    # Auto-detect
    provider = setup_default_model(api_key=api_key)
    _model_provider = provider
    set_provider(provider)
    print(f"Auto-detected model: {provider.config.provider_type}:{provider.config.model_id}")
    return provider


def list_cookbooks():
    """List all available cookbooks."""
    cookbooks = discover_cookbooks()
    
    print_header("Available Cookbooks")
    print(f"\nFound {len(cookbooks)} cookbooks:\n")
    
    by_tag = {}
    for cb in cookbooks:
        for tag in cb.tags or ["other"]:
            if tag not in by_tag:
                by_tag[tag] = []
            by_tag[tag].append(cb)
    
    for tag in sorted(by_tag.keys()):
        cbs = by_tag[tag]
        print(f"\n[{tag}] ({len(cbs)} cookbooks)")
        for cb in cbs:
            adapter_status = "✓" if cb.has_adapter else "○" if cb.has_manifest else "·"
            print(f"  {adapter_status} {cb.name}: {cb.title[:50]}")
    
    print("\n" + "=" * 70)
    print("Legend: ✓ has adapter.py  ○ has manifest  · auto-generated")
    print("=" * 70)


def list_models():
    """List available model configurations."""
    print_header("Model Provider Options")
    
    print("\nFormat: --model <provider>:<model_id>[:api_base]")
    
    print("\n[OpenAI API]")
    print("  --model openai:gpt-4o              # GPT-4o (default, vision)")
    print("  --model openai:gpt-4o-mini         # GPT-4o Mini (vision)")
    print("  --model openai:gpt-4-turbo         # GPT-4 Turbo (vision)")
    print("  --model openai:o1-preview          # O1 Preview (thinking)")
    print("  --model openai:o1-mini             # O1 Mini (thinking)")
    
    print("\n[Anthropic API]")
    print("  --model anthropic:claude-3-5-sonnet-20241022  # Claude 3.5 Sonnet (vision)")
    print("  --model anthropic:claude-3-7-sonnet-20250219  # Claude 3.7 (thinking)")
    print("  --model anthropic:claude-3-opus-20240229      # Claude 3 Opus (vision)")
    print("  --model anthropic:claude-3-haiku-20240307     # Claude 3 Haiku (fast)")
    
    print("\n[Ollama (Local)]")
    print("  --model ollama:llama3.2            # Llama 3.2")
    print("  --model ollama:llama3.2-vision     # Llama 3.2 Vision")
    print("  --model ollama:mistral             # Mistral")
    print("  --model ollama:qwen2.5             # Qwen 2.5")
    print("  --model ollama:deepseek-r1         # DeepSeek R1 (thinking)")
    
    print("\n[Transformers (Local HuggingFace)]")
    print("  --model transformers:meta-llama/Llama-3.2-3B-Instruct")
    print("  --model transformers:Qwen/Qwen2.5-7B-Instruct")
    print("  --model transformers:deepseek-ai/DeepSeek-R1-Distill-Qwen-7B  # thinking")
    print("  --model transformers:Qwen/Qwen2-VL-7B-Instruct  # Vision")
    
    print("\n[vLLM (Optimized Local)]")
    print("  --model vllm:mistralai/Mistral-7B-Instruct-v0.3")
    print("  --model vllm:meta-llama/Llama-3.2-3B-Instruct")
    
    print("\n[vLLM Server (Remote)]")
    print("  --model vllm-server:http://localhost:8000:model_name")
    
    print("\n" + "=" * 70)
    print("Model Capabilities:")
    print("  TEXT      - Standard text generation")
    print("  VISION    - Image understanding (VL models)")
    print("  THINKING  - Extended reasoning (o1, Claude 3.7, DeepSeek-R1)")
    print("  STREAMING - Stream responses")
    print("=" * 70)


def run_with_selection():
    """Run with interactive cookbook selection."""
    cookbooks = discover_cookbooks()
    
    if not cookbooks:
        print("No cookbooks found!")
        return
    
    selected = cookbook_selection_ui(cookbooks)
    
    if not selected:
        print("No cookbooks selected. Exiting.")
        return
    
    print(f"\nLoading {len(selected)} cookbooks...")
    registry = load_adapters(selected)
    
    adapters = registry.list_adapters()
    total_actions = sum(len(a.actions) for a in adapters)
    
    print(f"\nLoaded {len(adapters)} adapters with {total_actions} total actions")
    
    shared = {}
    if _model_provider:
        from model_providers import get_model_registry
        shared["model_registry"] = get_model_registry()
    
    run_interactive_agent(registry, shared)


def run_quick(question: str):
    """Run with default cookbooks."""
    cookbooks = discover_cookbooks()
    selected = quick_select_ui(cookbooks)
    
    print(f"Using {len(selected)} default cookbooks...")
    print(f"Question: {question}\n")
    
    try:
        answer = run_agent(question, selected)
        print("\n" + "=" * 70)
        print("ANSWER:")
        print("=" * 70)
        print(answer)
        print("=" * 70)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def run_with_cookbooks(cookbook_spec: str, question: str):
    """Run with specified cookbooks."""
    if "," in cookbook_spec:
        names = [n.strip() for n in cookbook_spec.split(",")]
    else:
        names = [cookbook_spec.strip()]
    
    cookbooks = discover_cookbooks()
    selected = select_by_names(cookbooks, names)
    
    if not selected:
        print(f"No cookbooks found matching: {cookbook_spec}")
        print("Available cookbooks:")
        for cb in cookbooks[:10]:
            print(f"  - {cb.name}")
        return
    
    print(f"Using cookbooks: {', '.join(selected)}")
    print(f"Question: {question}\n")
    
    try:
        answer = run_agent(question, selected)
        print("\n" + "=" * 70)
        print("ANSWER:")
        print("=" * 70)
        print(answer)
        print("=" * 70)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def run_with_tags(tags_spec: str, question: str):
    """Run with cookbooks matching specified tags."""
    tags = [t.strip() for t in tags_spec.split(",")]
    
    cookbooks = discover_cookbooks()
    selected = select_by_tags(cookbooks, tags)
    
    if not selected:
        print(f"No cookbooks found with tags: {tags_spec}")
        return
    
    print(f"Using {len(selected)} cookbooks with tags [{tags_spec}]")
    
    try:
        answer = run_agent(question, selected)
        print("\n" + "=" * 70)
        print("ANSWER:")
        print("=" * 70)
        print(answer)
        print("=" * 70)
    except Exception as e:
        print(f"Error: {e}")


def show_actions(cookbook_spec: str = None):
    """Show available actions for specified or all cookbooks."""
    cookbooks = discover_cookbooks()
    
    if cookbook_spec:
        names = [n.strip() for n in cookbook_spec.split(",")]
        selected = select_by_names(cookbooks, names)
    else:
        selected = [cb.name for cb in cookbooks]
    
    registry = load_adapters(selected)
    print_action_list(registry)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Unified Agent - Use any PocketFlow cookbook capability",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py                              # Interactive selection UI
    python main.py --list                       # List all cookbooks
    python main.py --quick "What is 2+2?"       # Quick run with defaults
    python main.py --select agent,rag "Query"   # Use specific cookbooks
    python main.py --tags agent,search "Query"  # Use cookbooks by tags
    python main.py --actions                    # Show all available actions
    
Model Selection:
    python main.py --model openai:gpt-4o "Query"
    python main.py --model ollama:llama3.2 --quick "Query"
    python main.py --model transformers:meta-llama/Llama-3.2-3B-Instruct "Query"
    python main.py --models                     # List model options
        """
    )
    
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all available cookbooks"
    )
    
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode after selecting cookbooks"
    )
    
    parser.add_argument(
        "--quick", "-q",
        metavar="QUESTION",
        help="Quick run with default cookbooks"
    )
    
    parser.add_argument(
        "--select", "-s",
        metavar="COOKBOOKS",
        help="Comma-separated list of cookbook names to use"
    )
    
    parser.add_argument(
        "--tags", "-t",
        metavar="TAGS",
        help="Use cookbooks matching these tags"
    )
    
    parser.add_argument(
        "--actions", "-a",
        nargs="?",
        const="",
        metavar="COOKBOOKS",
        help="Show available actions (optionally for specific cookbooks)"
    )
    
    parser.add_argument(
        "--model", "-m",
        metavar="MODEL_SPEC",
        help="Model specification (e.g., openai:gpt-4o, ollama:llama3.2)"
    )
    
    parser.add_argument(
        "--models",
        action="store_true",
        help="List available model options"
    )
    
    parser.add_argument(
        "--api-key",
        metavar="KEY",
        help="API key for the model provider"
    )
    
    parser.add_argument(
        "question",
        nargs="?",
        help="Question to ask (used with --select or --tags)"
    )
    
    args = parser.parse_args()
    
    if args.models:
        list_models()
        return
    
    # Check API key
    if not args.model and not os.environ.get("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set")
        print("   Set with: export OPENAI_API_KEY='your-key'")
        print("   Or use: --model ollama:llama3.2 for local models\n")
    
    # Setup model if specified
    if args.model:
        try:
            setup_model(args.model, args.api_key)
        except Exception as e:
            print(f"Error setting up model: {e}")
            return
    elif args.api_key:
        try:
            setup_model(api_key=args.api_key)
        except Exception as e:
            print(f"Error setting up model: {e}")
            return
    
    # Handle different modes
    if args.list:
        list_cookbooks()
    
    elif args.actions is not None:
        show_actions(args.actions if args.actions else None)
    
    elif args.quick:
        run_quick(args.quick)
    
    elif args.select and args.question:
        run_with_cookbooks(args.select, args.question)
    
    elif args.tags and args.question:
        run_with_tags(args.tags, args.question)
    
    elif args.interactive or (not args.select and not args.tags and not args.question):
        run_with_selection()
    
    elif args.question:
        run_quick(args.question)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
