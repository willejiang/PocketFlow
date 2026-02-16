"""
LLM utilities that use the model provider system.
"""

import os
from typing import List, Optional

_provider = None


def _get_provider():
    """Get or create the default model provider."""
    global _provider
    
    if _provider is not None:
        return _provider
    
    # Check if model registry has an active provider
    try:
        import sys
        parent_dir = os.path.dirname(os.path.dirname(__file__))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        
        from model_providers import get_model_registry
        
        registry = get_model_registry()
        if registry.active_name:
            _provider = registry.get_or_create_provider()
            return _provider
    except Exception:
        pass
    
    # Fallback to OpenAI
    try:
        from model_providers import ModelConfig, create_provider
        
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            config = ModelConfig.openai(api_key=api_key)
            _provider = create_provider(config)
            _provider.initialize()
            return _provider
    except Exception:
        pass
    
    # Legacy fallback
    return None


def set_provider(provider) -> None:
    """Set the global provider."""
    global _provider
    _provider = provider


def call_llm(
    prompt: str,
    model: str = None,
    max_retries: int = 3,
    temperature: float = 0.7,
    system_prompt: Optional[str] = None,
    **kwargs
) -> str:
    """Call LLM with the configured model provider."""
    if not prompt:
        raise ValueError("Prompt cannot be empty")
    
    provider = _get_provider()
    
    if provider is not None:
        return provider.chat(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_retries=max_retries,
            **kwargs
        )
    
    # Legacy OpenAI fallback
    from openai import OpenAI
    import time
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set and no model provider configured")
    
    client = OpenAI(api_key=api_key)
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model or "gpt-4o",
                messages=messages,
                temperature=temperature
            )
            if not response.choices:
                raise ValueError("Empty response")
            return response.choices[0].message.content
        except Exception as e:
            last_error = e
            time.sleep(1.0 * (2 ** attempt))
    
    raise last_error or Exception("Max retries exceeded")


def call_llm_with_image(
    prompt: str,
    image_path: Optional[str] = None,
    image_url: Optional[str] = None,
    image_base64: Optional[str] = None,
    system_prompt: Optional[str] = None,
    **kwargs
) -> str:
    """Call LLM with an image (vision model required)."""
    provider = _get_provider()
    
    if provider is not None:
        return provider.chat_with_image(
            prompt=prompt,
            image_path=image_path,
            image_url=image_url,
            image_base64=image_base64,
            system_prompt=system_prompt,
            **kwargs
        )
    
    raise ValueError("Vision support requires a configured model provider")


def get_embedding(text: str, model: str = "text-embedding-3-small") -> List[float]:
    """Get embedding for text."""
    if not text:
        raise ValueError("Text cannot be empty")
    
    from openai import OpenAI
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")
    
    client = OpenAI(api_key=api_key)
    
    if len(text) > 8000:
        text = text[:8000]
    
    response = client.embeddings.create(model=model, input=text)
    return response.data[0].embedding
