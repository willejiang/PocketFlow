"""
LLM calling utilities with retry logic and error handling.
"""

import os
import time
import json
from typing import Optional, Dict, Any, List

# Try to import OpenAI, but don't fail if not available
_openai_client = None
_openai_available = False

try:
    from openai import OpenAI, APIError, RateLimitError, APIConnectionError
    _openai_available = True
except ImportError:
    pass


class LLMError(Exception):
    """Base exception for LLM errors."""
    pass


class LLMConnectionError(LLMError):
    """Raised when connection to LLM fails."""
    pass


class LLMRateLimitError(LLMError):
    """Raised when rate limited."""
    pass


class LLMResponseError(LLMError):
    """Raised when response is invalid."""
    pass


def _get_client() -> "OpenAI":
    """Get or create OpenAI client with validation."""
    global _openai_client
    
    if not _openai_available:
        raise LLMError("OpenAI library not installed. Run: pip install openai")
    
    if _openai_client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise LLMError("OPENAI_API_KEY environment variable not set")
        
        if len(api_key) < 20:
            raise LLMError("OPENAI_API_KEY appears invalid (too short)")
        
        _openai_client = OpenAI(api_key=api_key)
    
    return _openai_client


def call_llm(
    prompt: str,
    model: str = "gpt-4o",
    max_retries: int = 3,
    retry_delay: float = 1.0,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    system_prompt: Optional[str] = None,
    timeout: float = 60.0
) -> str:
    """
    Call LLM with retry logic and error handling.
    
    Args:
        prompt: The user prompt
        model: Model to use
        max_retries: Maximum retry attempts
        retry_delay: Delay between retries (exponential backoff)
        temperature: Sampling temperature
        max_tokens: Maximum response tokens
        system_prompt: Optional system message
        timeout: Request timeout in seconds
        
    Returns:
        The LLM response text
        
    Raises:
        LLMError: On persistent failures
    """
    if not prompt or not prompt.strip():
        raise LLMError("Prompt cannot be empty")
    
    client = _get_client()
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    last_error = None
    
    for attempt in range(max_retries):
        try:
            kwargs = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "timeout": timeout
            }
            if max_tokens:
                kwargs["max_tokens"] = max_tokens
            
            response = client.chat.completions.create(**kwargs)
            
            if not response.choices:
                raise LLMResponseError("Empty response from LLM")
            
            content = response.choices[0].message.content
            if content is None:
                raise LLMResponseError("Null content in LLM response")
            
            return content
            
        except RateLimitError as e:
            last_error = LLMRateLimitError(f"Rate limited: {e}")
            delay = retry_delay * (2 ** attempt)
            time.sleep(delay)
            
        except APIConnectionError as e:
            last_error = LLMConnectionError(f"Connection failed: {e}")
            delay = retry_delay * (2 ** attempt)
            time.sleep(delay)
            
        except APIError as e:
            last_error = LLMError(f"API error: {e}")
            if "invalid_api_key" in str(e).lower():
                raise LLMError("Invalid API key") from e
            delay = retry_delay * (2 ** attempt)
            time.sleep(delay)
            
        except Exception as e:
            last_error = LLMError(f"Unexpected error: {type(e).__name__}: {e}")
            delay = retry_delay * (2 ** attempt)
            time.sleep(delay)
    
    raise last_error or LLMError("Max retries exceeded")


def call_llm_json(
    prompt: str,
    model: str = "gpt-4o",
    max_retries: int = 3,
    **kwargs
) -> Dict[str, Any]:
    """
    Call LLM and parse response as JSON.
    
    Automatically adds JSON instruction to prompt and handles parsing.
    """
    json_prompt = prompt + "\n\nRespond with valid JSON only, no markdown formatting."
    
    response = call_llm(json_prompt, model=model, max_retries=max_retries, **kwargs)
    
    # Try to extract JSON from response
    text = response.strip()
    
    # Handle markdown code blocks
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise LLMResponseError(f"Failed to parse JSON response: {e}\nResponse: {text[:500]}")


def get_embedding(
    text: str,
    model: str = "text-embedding-3-small",
    max_retries: int = 3,
    retry_delay: float = 1.0
) -> List[float]:
    """
    Get embedding for text.
    
    Args:
        text: Text to embed
        model: Embedding model
        max_retries: Retry attempts
        retry_delay: Delay between retries
        
    Returns:
        Embedding vector as list of floats
    """
    if not text or not text.strip():
        raise LLMError("Text cannot be empty")
    
    client = _get_client()
    
    # Truncate very long text
    max_chars = 8000
    if len(text) > max_chars:
        text = text[:max_chars]
    
    last_error = None
    
    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(
                model=model,
                input=text
            )
            
            if not response.data:
                raise LLMResponseError("Empty embedding response")
            
            return response.data[0].embedding
            
        except Exception as e:
            last_error = LLMError(f"Embedding failed: {e}")
            delay = retry_delay * (2 ** attempt)
            time.sleep(delay)
    
    raise last_error or LLMError("Max retries exceeded for embedding")


# Mock implementation for testing without API
class MockLLM:
    """Mock LLM for testing without API calls."""
    
    def __init__(self):
        self.calls = []
        self.responses = {}
    
    def set_response(self, pattern: str, response: str):
        """Set a canned response for prompts containing pattern."""
        self.responses[pattern] = response
    
    def __call__(self, prompt: str, **kwargs) -> str:
        self.calls.append({"prompt": prompt, "kwargs": kwargs})
        
        for pattern, response in self.responses.items():
            if pattern in prompt:
                return response
        
        return '```yaml\naction: answer\nreason: mock response\n```'


_mock_llm: Optional[MockLLM] = None


def enable_mock_llm() -> MockLLM:
    """Enable mock LLM for testing."""
    global _mock_llm
    _mock_llm = MockLLM()
    return _mock_llm


def disable_mock_llm():
    """Disable mock LLM."""
    global _mock_llm
    _mock_llm = None


def call_llm_or_mock(prompt: str, **kwargs) -> str:
    """Call LLM or mock if enabled."""
    if _mock_llm is not None:
        return _mock_llm(prompt, **kwargs)
    return call_llm(prompt, **kwargs)
