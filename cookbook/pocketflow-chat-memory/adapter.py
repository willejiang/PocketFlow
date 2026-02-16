"""
Adapter for pocketflow-chat-memory cookbook.

Provides conversation memory with vector-based retrieval.
"""

import sys
import os
from typing import Dict, Any, List

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pocketflow-unified-agent'))

from adapters.base import CookbookAdapter, AdapterAction


class ChatMemoryAdapter(CookbookAdapter):
    """Adapter for the chat memory cookbook."""
    
    def __init__(self):
        super().__init__()
        self._messages = []
        self._vector_index = None
        self._vector_items = []
    
    @property
    def name(self) -> str:
        return "pocketflow-chat-memory"
    
    @property
    def description(self) -> str:
        return "Conversation with long-term memory using vector retrieval"
    
    @property
    def tags(self) -> List[str]:
        return ["chat", "memory", "rag"]
    
    @property
    def dependencies(self) -> List[str]:
        return ["openai", "faiss-cpu", "numpy"]
    
    @property
    def actions(self) -> List[AdapterAction]:
        return [
            AdapterAction(
                name="chat_with_memory",
                description="Send a message and get a response using conversation memory",
                parameters={
                    "message": {"type": "str", "description": "User message", "required": True}
                }
            ),
            AdapterAction(
                name="recall_conversation",
                description="Recall past conversations relevant to a topic",
                parameters={
                    "topic": {"type": "str", "description": "Topic to recall", "required": True}
                }
            ),
            AdapterAction(
                name="clear_memory",
                description="Clear conversation memory",
                parameters={}
            )
        ]
    
    def initialize(self, shared: Dict[str, Any]) -> None:
        # Load memory from shared if available
        self._messages = shared.get("chat_messages", [])
        self._vector_index = shared.get("chat_vector_index")
        self._vector_items = shared.get("chat_vector_items", [])
        self._initialized = True
    
    def cleanup(self, shared: Dict[str, Any]) -> None:
        # Save memory to shared
        shared["chat_messages"] = self._messages
        shared["chat_vector_index"] = self._vector_index
        shared["chat_vector_items"] = self._vector_items
    
    def execute(
        self,
        action_name: str,
        params: Dict[str, Any],
        shared: Dict[str, Any]
    ) -> Dict[str, Any]:
        if action_name == "chat_with_memory":
            return self._chat(params, shared)
        elif action_name == "recall_conversation":
            return self._recall(params, shared)
        elif action_name == "clear_memory":
            return self._clear(shared)
        
        return {"success": False, "error": f"Unknown action: {action_name}"}
    
    def _chat(self, params: Dict[str, Any], shared: Dict[str, Any]) -> Dict[str, Any]:
        """Chat with memory."""
        message = params.get("message", "")
        
        if not message:
            return {"success": False, "error": "Message required"}
        
        try:
            try:
                from utils.call_llm import call_llm
            except ImportError:
                from utils import call_llm
            
            # Add user message
            self._messages.append({"role": "user", "content": message})
            
            # Try to recall relevant past conversations
            relevant = self._find_relevant(message)
            
            # Build context
            context_messages = []
            
            if relevant:
                context_messages.append({
                    "role": "system",
                    "content": "Relevant past conversations:\n" + "\n".join(relevant)
                })
            
            # Add recent messages (last 6)
            context_messages.extend(self._messages[-6:])
            
            # Format for LLM
            prompt = "\n".join([
                f"{m['role'].upper()}: {m['content']}"
                for m in context_messages
            ])
            prompt += "\nASSISTANT:"
            
            response = call_llm(prompt)
            
            # Add response to memory
            self._messages.append({"role": "assistant", "content": response})
            
            # Archive old messages to vector store
            if len(self._messages) > 6:
                self._archive_old_messages()
            
            # Save to shared
            shared["chat_messages"] = self._messages
            
            return {
                "success": True,
                "result": response,
                "context_update": f"Chat: {message}\nResponse: {response}"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _recall(self, params: Dict[str, Any], shared: Dict[str, Any]) -> Dict[str, Any]:
        """Recall past conversations."""
        topic = params.get("topic", "")
        
        if not topic:
            return {"success": False, "error": "Topic required"}
        
        relevant = self._find_relevant(topic)
        
        if relevant:
            return {
                "success": True,
                "result": relevant,
                "context_update": f"Recalled {len(relevant)} relevant conversations about '{topic}'"
            }
        else:
            return {
                "success": True,
                "result": [],
                "context_update": f"No past conversations about '{topic}'"
            }
    
    def _clear(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        """Clear memory."""
        self._messages = []
        self._vector_index = None
        self._vector_items = []
        
        shared.pop("chat_messages", None)
        shared.pop("chat_vector_index", None)
        shared.pop("chat_vector_items", None)
        
        return {
            "success": True,
            "result": "Memory cleared",
            "context_update": "Chat memory cleared"
        }
    
    def _find_relevant(self, query: str) -> List[str]:
        """Find relevant past conversations."""
        if not self._vector_items:
            return []
        
        try:
            from utils.get_embedding import get_embedding
            from utils.vector_index import search_vectors
            
            query_emb = get_embedding(query)
            indices, _ = search_vectors(self._vector_index, query_emb, k=2)
            
            return [self._vector_items[i] for i in indices if i < len(self._vector_items)]
            
        except ImportError:
            # Simple keyword matching fallback
            relevant = []
            query_lower = query.lower()
            for item in self._vector_items:
                if any(word in item.lower() for word in query_lower.split()):
                    relevant.append(item)
            return relevant[:2]
        except Exception:
            return []
    
    def _archive_old_messages(self):
        """Archive old messages to vector store."""
        if len(self._messages) <= 6:
            return
        
        # Get oldest pair
        old_pair = self._messages[:2]
        self._messages = self._messages[2:]
        
        # Format for storage
        text = f"User: {old_pair[0]['content']}\nAssistant: {old_pair[1]['content']}"
        
        try:
            from utils.get_embedding import get_embedding
            from utils.vector_index import create_index, add_vector
            
            if self._vector_index is None:
                self._vector_index = create_index()
            
            emb = get_embedding(text)
            add_vector(self._vector_index, emb)
            self._vector_items.append(text)
            
        except ImportError:
            # Just store without embeddings
            self._vector_items.append(text)


def get_adapter() -> CookbookAdapter:
    return ChatMemoryAdapter()
