"""Transformers model provider for local inference."""

import re
from typing import Any, Dict, List, Optional

from .base import (
    ModelProvider,
    ModelConfig,
    ModelCapability,
    ChatMessage,
    ModelResponse,
)


class TransformersProvider(ModelProvider):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._model = None
        self._tokenizer = None
        self._processor = None
    
    def initialize(self) -> None:
        if self._initialized:
            return
        
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
        except ImportError:
            raise ImportError(
                "transformers and torch packages not installed. "
                "Run: pip install transformers torch"
            )
        
        model_id = self.config.model_id
        device = self.config.device or "auto"
        dtype = self._resolve_dtype()
        
        load_kwargs = {
            "device_map": device if device != "cpu" else None,
            "torch_dtype": dtype,
            "trust_remote_code": True,
        }
        
        if self.config.extra_params:
            load_kwargs.update(self.config.extra_params)
        
        is_vl = self.config.is_vision_model
        
        if is_vl:
            try:
                self._processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            except Exception:
                self._tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        else:
            self._tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        
        self._model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        
        if device == "cpu":
            self._model = self._model.to("cpu")
        
        self._initialized = True
    
    def _resolve_dtype(self):
        import torch
        dtype_str = self.config.dtype or "auto"
        
        if dtype_str == "auto":
            if torch.cuda.is_available():
                return torch.float16
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.float16
            else:
                return torch.float32
        
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map.get(dtype_str, torch.float16)
    
    def generate(
        self,
        messages: List[ChatMessage],
        **kwargs
    ) -> ModelResponse:
        import torch
        
        if not self._initialized:
            self.initialize()
        
        has_image = any(
            isinstance(m.content, list) and any(p.get("type") == "image_url" for p in m.content)
            for m in messages
        )
        
        if has_image and self._processor:
            return self._generate_with_vision(messages, **kwargs)
        
        return self._generate_text(messages, **kwargs)
    
    def _generate_text(self, messages: List[ChatMessage], **kwargs) -> ModelResponse:
        import torch
        
        formatted = self._format_messages_for_chat(messages)
        
        if hasattr(self._tokenizer, "apply_chat_template"):
            prompt = self._tokenizer.apply_chat_template(
                formatted,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            prompt = self._fallback_format(formatted)
        
        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        
        gen_kwargs = {
            "max_new_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "do_sample": kwargs.get("temperature", self.config.temperature) > 0,
            "top_p": kwargs.get("top_p", self.config.top_p),
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        
        with torch.no_grad():
            outputs = self._model.generate(**inputs, **gen_kwargs)
        
        input_len = inputs["input_ids"].shape[1]
        generated = outputs[0][input_len:]
        content = self._tokenizer.decode(generated, skip_special_tokens=True)
        
        thinking = None
        if self.config.is_thinking_model:
            content, thinking = self._extract_thinking(content)
        
        return ModelResponse(
            content=content.strip(),
            thinking=thinking,
            usage={
                "prompt_tokens": input_len,
                "completion_tokens": len(generated),
                "total_tokens": input_len + len(generated),
            }
        )
    
    def _generate_with_vision(self, messages: List[ChatMessage], **kwargs) -> ModelResponse:
        import torch
        from PIL import Image
        import base64
        from io import BytesIO
        
        formatted = []
        images = []
        
        for msg in messages:
            if isinstance(msg.content, str):
                formatted.append({"role": msg.role, "content": msg.content})
            else:
                text_parts = []
                for part in msg.content:
                    if part.get("type") == "text":
                        text_parts.append(part["text"])
                    elif part.get("type") == "image_url":
                        url = part["image_url"]["url"]
                        if url.startswith("data:"):
                            b64_data = url.split(",", 1)[1]
                            img_bytes = base64.b64decode(b64_data)
                            img = Image.open(BytesIO(img_bytes))
                        else:
                            import requests
                            img_bytes = requests.get(url).content
                            img = Image.open(BytesIO(img_bytes))
                        images.append(img)
                        text_parts.append("<image>")
                
                formatted.append({"role": msg.role, "content": " ".join(text_parts)})
        
        if hasattr(self._processor, "apply_chat_template"):
            prompt = self._processor.apply_chat_template(formatted, tokenize=False, add_generation_prompt=True)
        else:
            prompt = self._fallback_format(formatted)
        
        if images:
            inputs = self._processor(text=prompt, images=images, return_tensors="pt")
        else:
            inputs = self._processor(text=prompt, return_tensors="pt")
        
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        
        gen_kwargs = {
            "max_new_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "do_sample": kwargs.get("temperature", self.config.temperature) > 0,
        }
        
        with torch.no_grad():
            outputs = self._model.generate(**inputs, **gen_kwargs)
        
        input_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
        generated = outputs[0][input_len:]
        
        if self._tokenizer:
            content = self._tokenizer.decode(generated, skip_special_tokens=True)
        else:
            content = self._processor.decode(generated, skip_special_tokens=True)
        
        return ModelResponse(content=content.strip())
    
    def _format_messages_for_chat(self, messages: List[ChatMessage]) -> List[Dict[str, str]]:
        formatted = []
        for msg in messages:
            if isinstance(msg.content, str):
                formatted.append({"role": msg.role, "content": msg.content})
            else:
                text_parts = [p["text"] for p in msg.content if p.get("type") == "text"]
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
    
    def _extract_thinking(self, content: str) -> tuple:
        patterns = [
            (r"<think>(.*?)</think>", ""),
            (r"<thinking>(.*?)</thinking>", ""),
            (r"\[thinking\](.*?)\[/thinking\]", ""),
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
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        
        self._initialized = False
