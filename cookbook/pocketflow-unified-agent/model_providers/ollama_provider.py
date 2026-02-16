"""Ollama model provider."""

import json
import time
from typing import Any, Dict, List, Optional

from .base import (
    ModelProvider,
    ModelConfig,
    ModelCapability,
    ChatMessage,
    ModelResponse,
)


class OllamaProvider(ModelProvider):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._base_url = config.api_base or "http://localhost:11434"
    
    def initialize(self) -> None:
        if self._initialized:
            return
        
        try:
            import requests
            resp = requests.get(f"{self._base_url}/api/tags", timeout=5)
            resp.raise_for_status()
        except ImportError:
            raise ImportError("requests package not installed")
        except Exception as e:
            raise ConnectionError(f"Cannot connect to Ollama at {self._base_url}: {e}")
        
        self._initialized = True
    
    def generate(
        self,
        messages: List[ChatMessage],
        max_retries: int = 3,
        **kwargs
    ) -> ModelResponse:
        import requests
        
        if not self._initialized:
            self.initialize()
        
        formatted = self._format_messages(messages)
        
        payload = {
            "model": self.config.model_id,
            "messages": formatted,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "num_predict": kwargs.get("max_tokens", self.config.max_tokens),
            }
        }
        
        if self.config.extra_params:
            payload["options"].update(self.config.extra_params)
        
        last_error = None
        for attempt in range(max_retries):
            try:
                resp = requests.post(
                    f"{self._base_url}/api/chat",
                    json=payload,
                    timeout=300
                )
                resp.raise_for_status()
                data = resp.json()
                return self._parse_response(data)
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    time.sleep(1.0 * (2 ** attempt))
        
        raise last_error
    
    def _format_messages(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        formatted = []
        for msg in messages:
            entry = {"role": msg.role}
            
            if isinstance(msg.content, str):
                entry["content"] = msg.content
            else:
                text_parts = []
                images = []
                for part in msg.content:
                    if part.get("type") == "text":
                        text_parts.append(part["text"])
                    elif part.get("type") == "image_url":
                        url = part["image_url"]["url"]
                        if url.startswith("data:"):
                            b64 = url.split(",", 1)[1]
                            images.append(b64)
                        else:
                            images.append(url)
                
                entry["content"] = " ".join(text_parts)
                if images:
                    entry["images"] = images
            
            formatted.append(entry)
        return formatted
    
    def _parse_response(self, data: Dict[str, Any]) -> ModelResponse:
        message = data.get("message", {})
        content = message.get("content", "")
        
        thinking = None
        if "<think>" in content:
            parts = content.split("</think>", 1)
            if len(parts) == 2:
                thinking = parts[0].replace("<think>", "").strip()
                content = parts[1].strip()
        
        usage = None
        if "eval_count" in data:
            usage = {
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
                "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
            }
        
        return ModelResponse(
            content=content,
            thinking=thinking,
            usage=usage,
            raw_response=data,
            finish_reason=data.get("done_reason"),
        )
