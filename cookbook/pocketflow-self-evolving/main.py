"""
Self-Evolving Agent - Main Entry Point

An agent that can create, register, and use tools dynamically,
expanding its capabilities over time.

Usage:
    python main.py --"What is the factorial of 10?"
    python main.py --list-tools
    python main.py --demo
"""

import sys
import os

# Add parent path for pocketflow import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from flow import create_self_evolving_agent_flow, create_tool_management_flow


def run_agent(question: str, db_path: str = "tool_registry.db"):
    """Run the self-evolving agent with a question."""
    print("=" * 60)
    print("🤖 Self-Evolving Agent")
    print("=" * 60)
    print(f"Question: {question}")
    print("-" * 60)
    
    # Create flow and shared state
    agent_flow = create_self_evolving_agent_flow(enable_learning=True)
    
    shared = {
        "question": question,
        "tool_registry_path": db_path
    }
    
    # Run the agent
    try:
        agent_flow.run(shared)
    except KeyboardInterrupt:
        print("\n⚠️  Agent interrupted")
        return None
    except Exception as e:
        print(f"\n❌ Agent error: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    print("-" * 60)
    print("📝 Final Answer:")
    print(shared.get("answer", "No answer generated"))
    print("=" * 60)
    
    return shared.get("answer")


def list_tools(db_path: str = "tool_registry.db"):
    """List all registered tools."""
    print("=" * 60)
    print("📋 Registered Tools")
    print("=" * 60)
    
    flow = create_tool_management_flow()
    shared = {"tool_registry_path": db_path}
    
    flow.run(shared)
    
    tools = shared.get("tools_list", [])
    if not tools:
        print("No tools registered yet.")
    
    print("=" * 60)
    return tools


def run_demo():
    """Run a demonstration of the self-evolving agent."""
    print("\n" + "=" * 60)
    print("🎮 Self-Evolving Agent Demo")
    print("=" * 60)
    
    # Use a temporary database for demo
    demo_db = "/tmp/demo_tool_registry.db"
    
    # Clean up any existing demo database
    if os.path.exists(demo_db):
        os.remove(demo_db)
    if os.path.exists(demo_db + ".lock"):
        os.remove(demo_db + ".lock")
    
    # Demo questions that might trigger tool creation
    demo_questions = [
        "Calculate the greatest common divisor of 48 and 18",
        "What's the GCD of 100 and 35?",  # Should reuse the tool if created
        "Convert 100 Fahrenheit to Celsius",
    ]
    
    print("\nThis demo will show how the agent:")
    print("1. Recognizes when a reusable tool could help")
    print("2. Creates and registers new tools")
    print("3. Reuses tools for similar problems")
    print("\n")
    
    for i, question in enumerate(demo_questions, 1):
        print(f"\n{'='*60}")
        print(f"Demo Question {i}/{len(demo_questions)}")
        print(f"{'='*60}")
        
        run_agent(question, db_path=demo_db)
        
        print("\nCurrent tools in registry:")
        list_tools(db_path=demo_db)
        
        if i < len(demo_questions):
            print("\n--- Continuing to next question ---\n")
    
    # Cleanup
    if os.path.exists(demo_db):
        os.remove(demo_db)
    if os.path.exists(demo_db + ".lock"):
        os.remove(demo_db + ".lock")


def print_usage():
    """Print usage information."""
    print("""
Self-Evolving Agent - Usage:

  Run with a question:
    python main.py --"Your question here"
    
  List registered tools:
    python main.py --list-tools
    
  Run demo:
    python main.py --demo
    
  Show help:
    python main.py --help

Environment Variables:
  OPENAI_API_KEY    - Required for LLM calls
  TOOL_REGISTRY_DB  - Path to tool registry database (default: tool_registry.db)
    """)


def main():
    """Main entry point."""
    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not set. LLM calls will fail.")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        print()
    
    # Get database path from environment or default
    db_path = os.environ.get("TOOL_REGISTRY_DB", "tool_registry.db")
    
    # Parse arguments
    if len(sys.argv) < 2:
        print_usage()
        return
    
    arg = sys.argv[1]
    
    if arg == "--help" or arg == "-h":
        print_usage()
        
    elif arg == "--list-tools":
        list_tools(db_path=db_path)
        
    elif arg == "--demo":
        run_demo()
        
    elif arg.startswith("--"):
        # Question mode
        question = arg[2:]
        if not question:
            print("Error: Empty question")
            print_usage()
            return
        run_agent(question, db_path=db_path)
        
    else:
        print(f"Unknown argument: {arg}")
        print_usage()


if __name__ == "__main__":
    main()
