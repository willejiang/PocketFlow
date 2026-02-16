"""Tests for model provider system."""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from model_providers import (
    ModelProvider,
    ModelConfig,
    ModelCapability,
    ChatMessage,
    ModelResponse,
    ModelRegistry,
    get_model_registry,
    create_provider,
)
from model_providers.factory import create_from_string


class TestModelConfig:
    def test_openai_config(self):
        config = ModelConfig.openai(model_id="gpt-4o")
        assert config.provider_type == "openai"
        assert config.model_id == "gpt-4o"
        assert ModelCapability.TEXT in config.capabilities
        assert ModelCapability.VISION in config.capabilities
    
    def test_openai_thinking_model(self):
        config = ModelConfig.openai(model_id="o1-preview")
        assert config.is_thinking_model
        assert ModelCapability.THINKING in config.capabilities
    
    def test_anthropic_config(self):
        config = ModelConfig.anthropic(model_id="claude-3-5-sonnet-20241022")
        assert config.provider_type == "anthropic"
        assert ModelCapability.VISION in config.capabilities
    
    def test_anthropic_thinking(self):
        config = ModelConfig.anthropic(model_id="claude-3-7-sonnet-20250219")
        assert config.is_thinking_model
    
    def test_ollama_config(self):
        config = ModelConfig.ollama(model_id="llama3.2")
        assert config.provider_type == "ollama"
        assert config.api_base == "http://localhost:11434"
    
    def test_ollama_vision(self):
        config = ModelConfig.ollama(model_id="llama3.2-vision")
        assert config.is_vision_model
    
    def test_transformers_config(self):
        config = ModelConfig.transformers(model_id="meta-llama/Llama-3.2-3B-Instruct")
        assert config.provider_type == "transformers"
        assert config.device == "auto"
    
    def test_transformers_thinking(self):
        config = ModelConfig.transformers(model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
        assert config.is_thinking_model
    
    def test_transformers_vision(self):
        config = ModelConfig.transformers(model_id="Qwen/Qwen2-VL-7B-Instruct")
        assert config.is_vision_model
    
    def test_vllm_config(self):
        config = ModelConfig.vllm(model_id="mistralai/Mistral-7B-Instruct-v0.3")
        assert config.provider_type == "vllm"


class TestCreateFromString:
    def test_simple_model_name(self):
        provider = create_from_string("gpt-4o")
        assert provider.config.provider_type == "openai"
        assert provider.config.model_id == "gpt-4o"
    
    def test_openai_spec(self):
        provider = create_from_string("openai:gpt-4-turbo")
        assert provider.config.provider_type == "openai"
        assert provider.config.model_id == "gpt-4-turbo"
    
    def test_anthropic_spec(self):
        provider = create_from_string("anthropic:claude-3-opus-20240229")
        assert provider.config.provider_type == "anthropic"
        assert provider.config.model_id == "claude-3-opus-20240229"
    
    def test_ollama_spec(self):
        provider = create_from_string("ollama:llama3.2")
        assert provider.config.provider_type == "ollama"
        assert provider.config.model_id == "llama3.2"
    
    def test_ollama_with_api_base(self):
        provider = create_from_string("ollama:llama3.2:http://192.168.1.100:11434")
        assert provider.config.model_id == "llama3.2"
        assert provider.config.api_base == "http://192.168.1.100:11434"
    
    def test_transformers_spec(self):
        provider = create_from_string("transformers:meta-llama/Llama-3.2-3B-Instruct")
        assert provider.config.provider_type == "transformers"
        assert provider.config.model_id == "meta-llama/Llama-3.2-3B-Instruct"
    
    def test_vllm_spec(self):
        provider = create_from_string("vllm:mistralai/Mistral-7B-Instruct-v0.3")
        assert provider.config.provider_type == "vllm"
    
    def test_vllm_server_spec(self):
        provider = create_from_string("vllm-server:http://localhost:8000:model_name")
        assert provider.config.provider_type == "vllm"
        assert provider.config.model_id == "model_name"
        assert provider.config.api_base == "http://localhost:8000"


class TestChatMessage:
    def test_user_message(self):
        msg = ChatMessage.user("Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
    
    def test_system_message(self):
        msg = ChatMessage.system("You are helpful")
        assert msg.role == "system"
        assert msg.content == "You are helpful"
    
    def test_assistant_message(self):
        msg = ChatMessage.assistant("Hi there!")
        assert msg.role == "assistant"
        assert msg.content == "Hi there!"
    
    def test_user_with_image_url(self):
        msg = ChatMessage.user_with_image("Describe this", image_url="https://example.com/image.jpg")
        assert msg.role == "user"
        assert isinstance(msg.content, list)
        assert msg.content[0]["type"] == "text"
        assert msg.content[1]["type"] == "image_url"


class TestModelRegistry:
    def setup_method(self):
        ModelRegistry.reset()
    
    def test_register_config(self):
        registry = get_model_registry()
        config = ModelConfig.openai(model_id="gpt-4o")
        registry.register_config("test", config)
        
        assert "test" in registry.list_configs()
    
    def test_set_active(self):
        registry = get_model_registry()
        registry.register_config("model1", ModelConfig.openai())
        registry.register_config("model2", ModelConfig.ollama())
        
        registry.set_active("model1")
        assert registry.active_name == "model1"
        
        registry.set_active("model2")
        assert registry.active_name == "model2"
    
    def test_get_model_info(self):
        registry = get_model_registry()
        registry.register_config("test", ModelConfig.openai(model_id="gpt-4o"))
        
        info = registry.get_model_info("test")
        assert info["model_id"] == "gpt-4o"
        assert info["provider_type"] == "openai"
    
    def test_reset(self):
        registry = get_model_registry()
        registry.register_config("test", ModelConfig.openai())
        
        ModelRegistry.reset()
        
        new_registry = get_model_registry()
        assert "test" not in new_registry.list_all()


class TestProviderCreation:
    def test_create_openai_provider(self):
        config = ModelConfig.openai(model_id="gpt-4o")
        provider = create_provider(config)
        assert provider.__class__.__name__ == "OpenAIProvider"
    
    def test_create_anthropic_provider(self):
        config = ModelConfig.anthropic()
        provider = create_provider(config)
        assert provider.__class__.__name__ == "AnthropicProvider"
    
    def test_create_ollama_provider(self):
        config = ModelConfig.ollama()
        provider = create_provider(config)
        assert provider.__class__.__name__ == "OllamaProvider"
    
    def test_create_transformers_provider(self):
        config = ModelConfig.transformers(model_id="test-model")
        provider = create_provider(config)
        assert provider.__class__.__name__ == "TransformersProvider"
    
    def test_create_vllm_provider(self):
        config = ModelConfig.vllm(model_id="test-model")
        provider = create_provider(config)
        assert provider.__class__.__name__ == "VLLMProvider"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
