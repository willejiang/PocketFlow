"""
Self-Evolving Agent Flow

Creates an agent that can:
1. Search existing tools
2. Create new reusable tools
3. Execute tools
4. Answer questions
5. Learn from problem-solving sessions

The flow forms a dynamic loop where the agent decides actions based on context.
"""

from pocketflow import Flow

from nodes import (
    DecideAction,
    SearchTools,
    CreateTool,
    UseTool,
    AnswerQuestion,
    ShouldCreateTool,
    ListTools
)


def create_self_evolving_agent_flow(enable_learning: bool = True) -> Flow:
    """
    Create the self-evolving agent flow.
    
    Flow structure:
    
        ┌─────────────────────────────────────────────────┐
        │                                                 │
        ▼                                                 │
    [DecideAction] ─── search_tools ──► [SearchTools] ───┘
        │                                                 
        ├── create_tool ──► [CreateTool] ────────────────┘
        │                                                 
        ├── use_tool ──► [UseTool] ──────────────────────┘
        │                                                 
        └── answer ──► [AnswerQuestion] ──► [ShouldCreateTool]?
                                                │
                                    create_tool │
                                                ▼
                                          [CreateTool]
    
    Args:
        enable_learning: If True, analyze completed sessions for tool creation opportunities
        
    Returns:
        Configured Flow instance
    """
    # Create node instances
    decide = DecideAction()
    search_tools = SearchTools()
    create_tool = CreateTool(max_retries=3, wait=1.0)
    use_tool = UseTool(max_retries=2, wait=0.5)
    answer = AnswerQuestion()
    
    # Connect decision node to action nodes
    decide - "search_tools" >> search_tools
    decide - "create_tool" >> create_tool
    decide - "use_tool" >> use_tool
    decide - "answer" >> answer
    
    # Action nodes loop back to decision
    search_tools - "decide" >> decide
    create_tool - "decide" >> decide
    use_tool - "decide" >> decide
    
    if enable_learning:
        # After answering, check if we should create a tool from learned patterns
        should_create = ShouldCreateTool()
        answer - "done" >> should_create
        
        # If learning suggests a tool, go create it
        # Use a separate CreateTool instance for learning path
        learn_create_tool = CreateTool(max_retries=3, wait=1.0)
        should_create - "create_tool" >> learn_create_tool
        # Learning-created tool is the end of the flow
    
    return Flow(start=decide)


def create_simple_agent_flow() -> Flow:
    """
    Create a simpler agent flow without the learning component.
    
    Useful for testing or when you don't want automatic tool creation.
    """
    return create_self_evolving_agent_flow(enable_learning=False)


def create_tool_management_flow() -> Flow:
    """
    Create a flow for managing tools (listing, etc.).
    """
    list_tools = ListTools()
    return Flow(start=list_tools)
