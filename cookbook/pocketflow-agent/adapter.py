"""
Adapter for pocketflow-agent cookbook.

Provides web search and research capabilities to the unified agent.
"""

import sys
import os
from typing import Dict, Any, List

# Add paths for imports
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pocketflow-unified-agent'))

from adapters.base import CookbookAdapter, AdapterAction


class AgentAdapter(CookbookAdapter):
    """Adapter for the research agent cookbook."""
    
    def __init__(self):
        super().__init__()
        self._search_fn = None
    
    @property
    def name(self) -> str:
        return "pocketflow-agent"
    
    @property
    def description(self) -> str:
        return "Research agent with web search capabilities using DuckDuckGo"
    
    @property
    def tags(self) -> List[str]:
        return ["agent", "search", "research"]
    
    @property
    def dependencies(self) -> List[str]:
        return ["openai", "duckduckgo-search", "pyyaml"]
    
    @property
    def actions(self) -> List[AdapterAction]:
        return [
            AdapterAction(
                name="search_web",
                description="Search the web using DuckDuckGo and return results",
                parameters={
                    "query": {
                        "type": "str",
                        "description": "Search query",
                        "required": True
                    },
                    "max_results": {
                        "type": "int",
                        "description": "Maximum number of results",
                        "required": False,
                        "default": 5
                    }
                }
            ),
            AdapterAction(
                name="research_question",
                description="Research a question by searching the web and synthesizing an answer",
                parameters={
                    "question": {
                        "type": "str",
                        "description": "Question to research",
                        "required": True
                    }
                }
            )
        ]
    
    def initialize(self, shared: Dict[str, Any]) -> None:
        """Initialize search function."""
        try:
            from utils import search_web_duckduckgo
            self._search_fn = search_web_duckduckgo
        except ImportError:
            # Fallback
            self._search_fn = self._fallback_search
        
        self._initialized = True
    
    def _fallback_search(self, query: str) -> str:
        """Fallback search using duckduckgo-search directly."""
        try:
            from duckduckgo_search import DDGS
            results = DDGS().text(query, max_results=5)
            return "\n\n".join([
                f"Title: {r['title']}\nURL: {r['href']}\nSnippet: {r['body']}"
                for r in results
            ])
        except Exception as e:
            return f"Search failed: {e}"
    
    def execute(
        self,
        action_name: str,
        params: Dict[str, Any],
        shared: Dict[str, Any]
    ) -> Dict[str, Any]:
        if action_name == "search_web":
            return self._execute_search(params)
        elif action_name == "research_question":
            return self._execute_research(params, shared)
        
        return {"success": False, "error": f"Unknown action: {action_name}"}
    
    def _execute_search(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute web search."""
        query = params.get("query", "")
        
        if not query:
            return {"success": False, "error": "Query cannot be empty"}
        
        try:
            if self._search_fn:
                results = self._search_fn(query)
            else:
                results = self._fallback_search(query)
            
            return {
                "success": True,
                "result": results,
                "context_update": f"Web search for '{query}':\n{results[:1000]}"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _execute_research(self, params: Dict[str, Any], shared: Dict[str, Any]) -> Dict[str, Any]:
        """Execute full research flow."""
        question = params.get("question", "")
        
        if not question:
            return {"success": False, "error": "Question cannot be empty"}
        
        try:
            from flow import create_agent_flow
            
            flow = create_agent_flow()
            flow_shared = {"question": question}
            flow.run(flow_shared)
            
            answer = flow_shared.get("answer", "No answer found")
            context = flow_shared.get("context", "")
            
            return {
                "success": True,
                "result": answer,
                "context_update": f"Research on '{question}':\n{context}\n\nAnswer: {answer}"
            }
        except Exception as e:
            # Fallback: just search
            search_result = self._execute_search({"query": question})
            return {
                "success": True,
                "result": search_result.get("result", ""),
                "context_update": f"Search results for '{question}':\n{search_result.get('result', '')}"
            }


def get_adapter() -> CookbookAdapter:
    return AgentAdapter()
