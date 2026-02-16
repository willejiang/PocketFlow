"""vLLM model provider - supports both local vLLM and vLLM server API."""

import time
from typing import Any, Dict, List, Optional

from .base import (
    ModelProvider,
    ModelConfig,
    ModelCapability,
    ChatMessage,
    ModelResponse,
)


class VLLMProvider(ModelProvider):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._llm = None
        self._use_server = config.api_base is not None
    
    def initialize(self) -> None:
        if self._initialized:
            return
        
        if self._use_server:
            self._initialize_client()
        else:
            self._initialize_local()
        
        self._initialized = True
    
    def _initialize_client(self) -> None:
        try:
            import requests
            resp = requests.get(f"{self.config.api_base}/health", timeout=5)
        except ImportError:
            raise ImportError("requests package not installed")
        except Exception as e:
            raise ConnectionError(f"Cannot connect to vLLM server at {self.config.api_base}: {e}")
    
    def _initialize_local(self) -> None:
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError(
                "vllm package not installed. "
                "Run: pip install vllm"
            )
        
        kwargs = {
            "model": self.config.model_id,
            "trust_remote_code": True,
        }
        
        if self.config.dtype:
            kwargs["dtype"] = self.config.dtype
        
        if self.config.extra_params:
            kwargs.update(self.config.extra_params)
        
        self._llm = LLM(**kwargs)
    
    def generate(
        self,
        messages: List[ChatMessage],
        max_retries: int = 3,
        **kwargs
    ) -> ModelResponse:
        if not self._initialized:
            self.initialize()
        
        if self._use_server:
            return self._generate_via_server(messages, max_retries, **kwargs)
        else:
            return self._generate_local(messages, **kwargs)
    
    def _generate_via_server(
        self,
        messages: List[ChatMessage],
        max_retries: int,
        **kwargs
    ) -> ModelResponse:
        import requests
        
        formatted = self._format_messages(messages)
        
        payload = {
            "model": self.config.model_id,
            "messages": formatted,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
        }
        
        last_error = None
        for attempt in range(max_retries):
            try:
                resp = requests.post(
                    f"{self.config.api_base}/v1/chat/completions",
                    json=payload,
                    timeout=300
                )
                resp.raise_for_status()
                data = resp.json()
                return self._parse_server_response(data)
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    time.sleep(1.0 * (2 ** attempt))
        
        raise last_error
    
    def _generate_local(self, messages: List[ChatMessage], **kwargs) -> ModelResponse:
        from vllm import SamplingParams
        
        formatted = self._format_messages(messages)
        
        tokenizer = self._llm.get_tokenizer()
        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(
                formatted, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = self._fallback_format(formatted)
        
        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            top_p=kwargs.get("top_p", self.config.top_p),
        )
        
        outputs = self._llm.generate([prompt], sampling_params)
        output = outputs[0]
        
        content = output.outputs[0].text
        
        thinking = None
        if self.config.is_thinking_model:
            content, thinking = self._extract_thinking(content)
        
        usage = {
            "prompt_tokens": len(output.prompt_token_ids),
            "completion_tokens": len(output.outputs[0].token_ids),
            "total_tokens": len(output.prompt_token_ids) + len(output.outputs[0].token_ids),
        }
        
        return ModelResponse(
            content=content.strip(),
            thinking=thinking,
            usage=usage,
            finish_reason=output.outputs[0].finish_reason,
        )
    
    def _format_messages(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        formatted = []
        for msg in messages:
            if isinstance(msg.content, str):
                formatted.append({"role": msg.role, "content": msg.content})
            else:
                text_parts = []
                for part in msg.content:
                    if part.get("type") == "text":
                        text_parts.append(part["text"])
                formatted.append({"role": msg.role, "content": " ".join(text_parts)})
        return formatted
    
    def _fallback_format(self, messages: List[Dict[str, str]]) -> str:
        parts = []
        for m in messages:
            role = m["role"]
            content = m["content"]
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
        parts.append("Assistant:")
        return "\n\n".join(parts)
    
    def _parse_server_response(self, data: Dict[str, Any]) -> ModelResponse:
        choice = data["choices"][0]
        content = choice["message"]["content"]
        
        thinking = None
        if self.config.is_thinking_model:
            content, thinking = self._extract_thinking(content)
        
        usage = None
        if "usage" in data:
            usage = data["usage"]
        
        return ModelResponse(
            content=content.strip(),
            thinking=thinking,
            usage=usage,
            raw_response=data,
            finish_reason=choice.get("finish_reason"),
        )
    
    def _extract_thinking(self, content: str) -> tuple:
        import re
        
        patterns = [
            (r"<think>(.*?)</think>", ""),
            (r"<thinking>(.*?)</thinking>", ""),
        ]
        
        thinking = None
        for pattern, _ in patterns:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                thinking = match.group(1).strip()
                content = re.sub(pattern, "", content, flags=re.DOTALL).strip()
                break
        
        return content, thinking
    
    def cleanup(self) -> None:
        if self._llm is not None:
            del self._llm
            self._llm = None
        
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        
        self._initialized = False
