"""Factory for creating model providers."""

from typing import Optional

from .base import ModelProvider, ModelConfig


def create_provider(config: ModelConfig) -> ModelProvider:
    provider_type = config.provider_type.lower()
    
    if provider_type == "openai":
        from .openai_provider import OpenAIProvider
        return OpenAIProvider(config)
    
    elif provider_type == "anthropic":
        from .anthropic_provider import AnthropicProvider
        return AnthropicProvider(config)
    
    elif provider_type == "ollama":
        from .ollama_provider import OllamaProvider
        return OllamaProvider(config)
    
    elif provider_type == "transformers":
        from .transformers_provider import TransformersProvider
        return TransformersProvider(config)
    
    elif provider_type == "vllm":
        from .vllm_provider import VLLMProvider
        return VLLMProvider(config)
    
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")


def create_from_string(model_spec: str) -> ModelProvider:
    """
    Create a provider from a string specification.
    
    Formats:
        "openai:gpt-4o"
        "ollama:llama3.2"
        "transformers:meta-llama/Llama-3.2-3B-Instruct"
        "vllm:mistralai/Mistral-7B-Instruct-v0.3"
        "vllm-server:http://localhost:8000/v1:model_name"
    """
    if ":" not in model_spec:
        config = ModelConfig.openai(model_id=model_spec)
        return create_provider(config)
    
    # Handle vllm-server specially due to URL containing colons
    if model_spec.lower().startswith("vllm-server:"):
        rest = model_spec[len("vllm-server:"):]
        # Find the last colon that separates URL from model name
        if rest.count(":") >= 2:
            # URL has port, format: http://host:port:model or http://host:port/path:model
            last_colon = rest.rfind(":")
            api_base = rest[:last_colon]
            model_id = rest[last_colon + 1:] if last_colon < len(rest) - 1 else "model"
        else:
            # No model name specified
            api_base = rest
            model_id = "model"
        
        config = ModelConfig.vllm(model_id=model_id, api_base=api_base)
        return create_provider(config)
    
    parts = model_spec.split(":", 1)
    provider_type = parts[0].lower()
    rest = parts[1] if len(parts) > 1 else ""
    
    if provider_type == "openai":
        model_id = rest or "gpt-4o"
        config = ModelConfig.openai(model_id=model_id)
    
    elif provider_type == "anthropic":
        model_id = rest or "claude-3-5-sonnet-20241022"
        config = ModelConfig.anthropic(model_id=model_id)
    
    elif provider_type == "ollama":
        # ollama:model or ollama:model:http://host:port
        if rest.count(":") >= 2:
            # Has custom API base
            idx = rest.find(":http")
            if idx > 0:
                model_id = rest[:idx]
                api_base = rest[idx + 1:]
            else:
                model_id = rest
                api_base = "http://localhost:11434"
        else:
            model_id = rest or "llama3.2"
            api_base = "http://localhost:11434"
        config = ModelConfig.ollama(model_id=model_id, api_base=api_base)
    
    elif provider_type == "transformers":
        model_id = rest or "meta-llama/Llama-3.2-3B-Instruct"
        config = ModelConfig.transformers(model_id=model_id)
    
    elif provider_type == "vllm":
        model_id = rest or "mistralai/Mistral-7B-Instruct-v0.3"
        config = ModelConfig.vllm(model_id=model_id, api_base=None)
    
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")
    
    return create_provider(config)


def setup_default_model(
    model_spec: Optional[str] = None,
    api_key: Optional[str] = None
) -> ModelProvider:
    """
    Setup the default model provider.
    
    If model_spec is None, tries to auto-detect available providers.
    """
    import os
    from .registry import get_model_registry
    
    registry = get_model_registry()
    
    if model_spec:
        provider = create_from_string(model_spec)
        if api_key and hasattr(provider.config, 'api_key'):
            provider.config.api_key = api_key
        provider.initialize()
        registry.register_provider("default", provider)
        registry.set_active("default")
        return provider
    
    # Auto-detect
    openai_key = api_key or os.environ.get("OPENAI_API_KEY")
    if openai_key:
        config = ModelConfig.openai(api_key=openai_key)
        provider = create_provider(config)
        provider.initialize()
        registry.register_provider("default", provider)
        registry.set_active("default")
        return provider
    
    try:
        import requests
        resp = requests.get("http://localhost:11434/api/tags", timeout=2)
        if resp.ok:
            config = ModelConfig.ollama()
            provider = create_provider(config)
            provider.initialize()
            registry.register_provider("default", provider)
            registry.set_active("default")
            return provider
    except Exception:
        pass
    
    raise ValueError(
        "No model provider available. Set OPENAI_API_KEY or run Ollama locally."
    )
