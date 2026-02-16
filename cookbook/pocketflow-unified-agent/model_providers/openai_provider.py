"""OpenAI model provider."""

import os
import time
from typing import Any, Dict, List, Optional

from .base import (
    ModelProvider,
    ModelConfig,
    ModelCapability,
    ChatMessage,
    ModelResponse,
)


class OpenAIProvider(ModelProvider):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._client = None
    
    def initialize(self) -> None:
        if self._initialized:
            return
        
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")
        
        api_key = self.config.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not provided")
        
        kwargs = {"api_key": api_key}
        if self.config.api_base:
            kwargs["base_url"] = self.config.api_base
        
        self._client = OpenAI(**kwargs)
        self._initialized = True
    
    def generate(
        self,
        messages: List[ChatMessage],
        max_retries: int = 3,
        **kwargs
    ) -> ModelResponse:
        if not self._initialized:
            self.initialize()
        
        formatted = self._format_messages(messages)
        
        params = {
            "model": self.config.model_id,
            "messages": formatted,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
        }
        
        if self.config.is_thinking_model:
            params.pop("temperature", None)
            params.pop("max_tokens", None)
            if "max_completion_tokens" not in kwargs:
                params["max_completion_tokens"] = self.config.max_tokens
        
        params.update(self.config.extra_params)
        params.update({k: v for k, v in kwargs.items() if k not in params})
        
        last_error = None
        for attempt in range(max_retries):
            try:
                response = self._client.chat.completions.create(**params)
                return self._parse_response(response)
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    time.sleep(1.0 * (2 ** attempt))
        
        raise last_error
    
    def _format_messages(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        formatted = []
        for msg in messages:
            if isinstance(msg.content, str):
                formatted.append({"role": msg.role, "content": msg.content})
            else:
                formatted.append({"role": msg.role, "content": msg.content})
        return formatted
    
    def _parse_response(self, response) -> ModelResponse:
        choice = response.choices[0]
        content = choice.message.content or ""
        
        thinking = None
        if self.config.is_thinking_model and hasattr(choice.message, "reasoning_content"):
            thinking = choice.message.reasoning_content
        
        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        
        return ModelResponse(
            content=content,
            thinking=thinking,
            usage=usage,
            raw_response=response,
            finish_reason=choice.finish_reason,
        )
