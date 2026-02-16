"""
Adapter for pocketflow-rag cookbook.

Provides RAG (Retrieval Augmented Generation) capabilities.
"""

import sys
import os
from typing import Dict, Any, List

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pocketflow-unified-agent'))

from adapters.base import CookbookAdapter, AdapterAction


class RAGAdapter(CookbookAdapter):
    """Adapter for the RAG cookbook."""
    
    def __init__(self):
        super().__init__()
        self._index = None
        self._texts = []
        self._embeddings = None
    
    @property
    def name(self) -> str:
        return "pocketflow-rag"
    
    @property
    def description(self) -> str:
        return "Retrieval Augmented Generation - Index documents and retrieve relevant context"
    
    @property
    def tags(self) -> List[str]:
        return ["rag", "retrieval", "embeddings"]
    
    @property
    def dependencies(self) -> List[str]:
        return ["openai", "faiss-cpu", "numpy"]
    
    @property
    def actions(self) -> List[AdapterAction]:
        return [
            AdapterAction(
                name="rag_index_documents",
                description="Index documents for later retrieval",
                parameters={
                    "documents": {
                        "type": "list",
                        "description": "List of document texts to index",
                        "required": True
                    }
                }
            ),
            AdapterAction(
                name="rag_retrieve",
                description="Retrieve relevant document chunks for a query",
                parameters={
                    "query": {
                        "type": "str",
                        "description": "Query to find relevant documents for",
                        "required": True
                    },
                    "top_k": {
                        "type": "int",
                        "description": "Number of documents to retrieve",
                        "required": False,
                        "default": 3
                    }
                }
            ),
            AdapterAction(
                name="rag_query",
                description="Query indexed documents and generate an answer",
                parameters={
                    "query": {
                        "type": "str",
                        "description": "Question to answer based on indexed documents",
                        "required": True
                    }
                }
            )
        ]
    
    def initialize(self, shared: Dict[str, Any]) -> None:
        # Load existing index from shared if available
        if "rag_index" in shared:
            self._index = shared["rag_index"]
            self._texts = shared.get("rag_texts", [])
        self._initialized = True
    
    def cleanup(self, shared: Dict[str, Any]) -> None:
        # Save index to shared
        if self._index is not None:
            shared["rag_index"] = self._index
            shared["rag_texts"] = self._texts
    
    def execute(
        self,
        action_name: str,
        params: Dict[str, Any],
        shared: Dict[str, Any]
    ) -> Dict[str, Any]:
        if action_name == "rag_index_documents":
            return self._index_documents(params, shared)
        elif action_name == "rag_retrieve":
            return self._retrieve(params, shared)
        elif action_name == "rag_query":
            return self._query(params, shared)
        
        return {"success": False, "error": f"Unknown action: {action_name}"}
    
    def _index_documents(self, params: Dict[str, Any], shared: Dict[str, Any]) -> Dict[str, Any]:
        """Index documents."""
        documents = params.get("documents", [])
        
        if not documents:
            return {"success": False, "error": "No documents provided"}
        
        try:
            from utils import get_embedding, fixed_size_chunk
            import numpy as np
            import faiss
            
            # Chunk documents
            all_chunks = []
            for doc in documents:
                chunks = fixed_size_chunk(doc)
                all_chunks.extend(chunks)
            
            self._texts = all_chunks
            
            # Get embeddings
            embeddings = [get_embedding(chunk) for chunk in all_chunks]
            self._embeddings = np.array(embeddings, dtype=np.float32)
            
            # Create index
            dimension = self._embeddings.shape[1]
            self._index = faiss.IndexFlatL2(dimension)
            self._index.add(self._embeddings)
            
            # Save to shared
            shared["rag_index"] = self._index
            shared["rag_texts"] = self._texts
            shared["rag_embeddings"] = self._embeddings
            
            return {
                "success": True,
                "result": f"Indexed {len(all_chunks)} chunks from {len(documents)} documents",
                "context_update": f"RAG: Indexed {len(all_chunks)} document chunks"
            }
            
        except ImportError as e:
            return {"success": False, "error": f"Missing dependency: {e}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _retrieve(self, params: Dict[str, Any], shared: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve relevant documents."""
        query = params.get("query", "")
        top_k = params.get("top_k", 3)
        
        if not query:
            return {"success": False, "error": "Query cannot be empty"}
        
        # Load from shared if needed
        if self._index is None:
            self._index = shared.get("rag_index")
            self._texts = shared.get("rag_texts", [])
        
        if self._index is None:
            return {"success": False, "error": "No documents indexed. Use rag_index_documents first."}
        
        try:
            from utils import get_embedding
            import numpy as np
            
            # Get query embedding
            query_emb = np.array([get_embedding(query)], dtype=np.float32)
            
            # Search
            distances, indices = self._index.search(query_emb, top_k)
            
            # Get results
            results = []
            for idx, dist in zip(indices[0], distances[0]):
                if idx < len(self._texts):
                    results.append({
                        "text": self._texts[idx],
                        "distance": float(dist)
                    })
            
            context = "\n\n".join([
                f"[{i+1}] {r['text']}" for i, r in enumerate(results)
            ])
            
            return {
                "success": True,
                "result": results,
                "context_update": f"Retrieved {len(results)} relevant documents:\n{context}"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _query(self, params: Dict[str, Any], shared: Dict[str, Any]) -> Dict[str, Any]:
        """Query documents and generate answer."""
        query = params.get("query", "")
        
        # First retrieve
        retrieve_result = self._retrieve({"query": query, "top_k": 3}, shared)
        
        if not retrieve_result.get("success"):
            return retrieve_result
        
        try:
            from utils import call_llm
            
            results = retrieve_result.get("result", [])
            context = "\n\n".join([r["text"] for r in results])
            
            prompt = f"""Answer the question based on the context provided.

Question: {query}

Context:
{context}

Answer:"""
            
            answer = call_llm(prompt)
            
            return {
                "success": True,
                "result": answer,
                "context_update": f"RAG Query: {query}\nAnswer: {answer}"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


def get_adapter() -> CookbookAdapter:
    return RAGAdapter()
