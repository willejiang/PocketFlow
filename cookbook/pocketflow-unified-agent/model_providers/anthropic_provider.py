"""Anthropic model provider."""

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


class AnthropicProvider(ModelProvider):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._client = None
    
    def initialize(self) -> None:
        if self._initialized:
            return
        
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")
        
        api_key = self.config.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key not provided")
        
        self._client = Anthropic(api_key=api_key)
        self._initialized = True
    
    def generate(
        self,
        messages: List[ChatMessage],
        max_retries: int = 3,
        **kwargs
    ) -> ModelResponse:
        if not self._initialized:
            self.initialize()
        
        system_prompt = None
        chat_messages = []
        
        for msg in messages:
            if msg.role == "system":
                system_prompt = msg.content if isinstance(msg.content, str) else str(msg.content)
            else:
                chat_messages.append(self._format_message(msg))
        
        params = {
            "model": self.config.model_id,
            "messages": chat_messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
        }
        
        if system_prompt:
            params["system"] = system_prompt
        
        # Only add temperature for non-thinking models
        if not self.config.is_thinking_model:
            params["temperature"] = kwargs.get("temperature", self.config.temperature)
        
        if self.config.extra_params:
            params.update(self.config.extra_params)
        
        last_error = None
        for attempt in range(max_retries):
            try:
                response = self._client.messages.create(**params)
                return self._parse_response(response)
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    time.sleep(1.0 * (2 ** attempt))
        
        raise last_error
    
    def _format_message(self, msg: ChatMessage) -> Dict[str, Any]:
        if isinstance(msg.content, str):
            return {"role": msg.role, "content": msg.content}
        
        # Handle multimodal content
        content = []
        for part in msg.content:
            if part.get("type") == "text":
                content.append({"type": "text", "text": part["text"]})
            elif part.get("type") == "image_url":
                url = part["image_url"]["url"]
                if url.startswith("data:"):
                    # Parse data URL
                    media_type = "image/jpeg"
                    if "image/png" in url:
                        media_type = "image/png"
                    elif "image/gif" in url:
                        media_type = "image/gif"
                    elif "image/webp" in url:
                        media_type = "image/webp"
                    
                    b64_data = url.split(",", 1)[1]
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64_data,
                        }
                    })
                else:
                    # URL-based image
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "url",
                            "url": url,
                        }
                    })
        
        return {"role": msg.role, "content": content}
    
    def _parse_response(self, response) -> ModelResponse:
        content_blocks = response.content
        
        text_content = ""
        thinking = None
        
        for block in content_blocks:
            if block.type == "text":
                text_content += block.text
            elif hasattr(block, "type") and block.type == "thinking":
                thinking = block.thinking
        
        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            }
        
        return ModelResponse(
            content=text_content,
            thinking=thinking,
            usage=usage,
            raw_response=response,
            finish_reason=response.stop_reason,
        )
