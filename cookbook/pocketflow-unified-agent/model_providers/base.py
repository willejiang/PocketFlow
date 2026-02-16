"""Base classes for model providers."""

import base64
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class ModelCapability(Enum):
    TEXT = auto()
    VISION = auto()
    THINKING = auto()
    FUNCTION_CALLING = auto()
    STREAMING = auto()


@dataclass
class ChatMessage:
    role: str  # "system", "user", "assistant"
    content: Union[str, List[Dict[str, Any]]]
    
    @classmethod
    def user(cls, content: str) -> "ChatMessage":
        return cls(role="user", content=content)
    
    @classmethod
    def system(cls, content: str) -> "ChatMessage":
        return cls(role="system", content=content)
    
    @classmethod
    def assistant(cls, content: str) -> "ChatMessage":
        return cls(role="assistant", content=content)
    
    @classmethod
    def user_with_image(
        cls, 
        text: str, 
        image_path: Optional[str] = None,
        image_url: Optional[str] = None,
        image_base64: Optional[str] = None
    ) -> "ChatMessage":
        content = []
        if text:
            content.append({"type": "text", "text": text})
        
        if image_path:
            path = Path(image_path)
            if path.exists():
                with open(path, "rb") as f:
                    img_data = base64.b64encode(f.read()).decode("utf-8")
                suffix = path.suffix.lower().lstrip(".")
                mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "gif": "gif", "webp": "webp"}.get(suffix, "jpeg")
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/{mime};base64,{img_data}"}
                })
        elif image_url:
            content.append({
                "type": "image_url",
                "image_url": {"url": image_url}
            })
        elif image_base64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
            })
        
        return cls(role="user", content=content)


@dataclass
class ModelResponse:
    content: str
    thinking: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    raw_response: Optional[Any] = None
    finish_reason: Optional[str] = None


@dataclass
class ModelConfig:
    provider_type: str  # "openai", "ollama", "transformers", "vllm"
    model_id: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    capabilities: List[ModelCapability] = field(default_factory=list)
    
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 1.0
    
    device: Optional[str] = None  # for transformers: "cuda", "cpu", "mps"
    dtype: Optional[str] = None  # "float16", "bfloat16", "float32"
    
    extra_params: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_thinking_model(self) -> bool:
        return ModelCapability.THINKING in self.capabilities
    
    @property
    def is_vision_model(self) -> bool:
        return ModelCapability.VISION in self.capabilities
    
    @classmethod
    def openai(
        cls,
        model_id: str = "gpt-4o",
        api_key: Optional[str] = None,
        **kwargs
    ) -> "ModelConfig":
        caps = [ModelCapability.TEXT, ModelCapability.STREAMING, ModelCapability.FUNCTION_CALLING]
        if "vision" in model_id or model_id in ("gpt-4o", "gpt-4o-mini", "gpt-4-turbo"):
            caps.append(ModelCapability.VISION)
        if "o1" in model_id or "o3" in model_id:
            caps.append(ModelCapability.THINKING)
        
        return cls(
            provider_type="openai",
            model_id=model_id,
            api_key=api_key,
            capabilities=caps,
            **kwargs
        )
    
    @classmethod
    def ollama(
        cls,
        model_id: str = "llama3.2",
        api_base: str = "http://localhost:11434",
        **kwargs
    ) -> "ModelConfig":
        caps = [ModelCapability.TEXT, ModelCapability.STREAMING]
        model_lower = model_id.lower()
        if any(v in model_lower for v in ("llava", "vision", "bakllava")):
            caps.append(ModelCapability.VISION)
        
        return cls(
            provider_type="ollama",
            model_id=model_id,
            api_base=api_base,
            capabilities=caps,
            **kwargs
        )
    
    @classmethod
    def transformers(
        cls,
        model_id: str,
        device: str = "auto",
        dtype: str = "auto",
        **kwargs
    ) -> "ModelConfig":
        caps = [ModelCapability.TEXT]
        model_lower = model_id.lower()
        if any(v in model_lower for v in ("llava", "vision", "vl", "qwen2-vl", "internvl")):
            caps.append(ModelCapability.VISION)
        if any(t in model_lower for t in ("deepseek-r1", "qwq", "skywork-or")):
            caps.append(ModelCapability.THINKING)
        
        return cls(
            provider_type="transformers",
            model_id=model_id,
            device=device,
            dtype=dtype,
            capabilities=caps,
            **kwargs
        )
    
    @classmethod
    def vllm(
        cls,
        model_id: str,
        api_base: str = "http://localhost:8000",
        **kwargs
    ) -> "ModelConfig":
        caps = [ModelCapability.TEXT, ModelCapability.STREAMING]
        model_lower = model_id.lower()
        if any(v in model_lower for v in ("llava", "vision", "vl")):
            caps.append(ModelCapability.VISION)
        if any(t in model_lower for t in ("deepseek-r1", "qwq")):
            caps.append(ModelCapability.THINKING)
        
        return cls(
            provider_type="vllm",
            model_id=model_id,
            api_base=api_base,
            capabilities=caps,
            **kwargs
        )
    
    @classmethod
    def anthropic(
        cls,
        model_id: str = "claude-3-5-sonnet-20241022",
        api_key: Optional[str] = None,
        **kwargs
    ) -> "ModelConfig":
        caps = [ModelCapability.TEXT, ModelCapability.STREAMING, ModelCapability.VISION]
        model_lower = model_id.lower()
        
        # Claude 3.7 sonnet has extended thinking
        if "3-7" in model_id or "3.7" in model_id:
            caps.append(ModelCapability.THINKING)
        
        return cls(
            provider_type="anthropic",
            model_id=model_id,
            api_key=api_key,
            capabilities=caps,
            **kwargs
        )


class ModelProvider(ABC):
    def __init__(self, config: ModelConfig):
        self.config = config
        self._initialized = False
    
    @abstractmethod
    def initialize(self) -> None:
        pass
    
    @abstractmethod
    def generate(
        self,
        messages: List[ChatMessage],
        **kwargs
    ) -> ModelResponse:
        pass
    
    def chat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        messages = []
        if system_prompt:
            messages.append(ChatMessage.system(system_prompt))
        messages.append(ChatMessage.user(prompt))
        
        response = self.generate(messages, **kwargs)
        return response.content
    
    def chat_with_image(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        image_url: Optional[str] = None,
        image_base64: Optional[str] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        if not self.config.is_vision_model:
            raise ValueError(f"Model {self.config.model_id} does not support vision")
        
        messages = []
        if system_prompt:
            messages.append(ChatMessage.system(system_prompt))
        messages.append(ChatMessage.user_with_image(
            prompt, image_path=image_path, image_url=image_url, image_base64=image_base64
        ))
        
        response = self.generate(messages, **kwargs)
        return response.content
    
    @property
    def capabilities(self) -> List[ModelCapability]:
        return self.config.capabilities
    
    def cleanup(self) -> None:
        pass
