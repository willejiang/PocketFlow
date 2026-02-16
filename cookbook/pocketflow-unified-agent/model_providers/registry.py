"""Model registry for managing model providers."""

import threading
from typing import Dict, List, Optional

from .base import ModelProvider, ModelConfig, ModelCapability


class ModelRegistry:
    _instance: Optional["ModelRegistry"] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._providers: Dict[str, ModelProvider] = {}
                cls._instance._active_provider: Optional[str] = None
                cls._instance._configs: Dict[str, ModelConfig] = {}
            return cls._instance
    
    def register_config(self, name: str, config: ModelConfig) -> None:
        with self._lock:
            self._configs[name] = config
    
    def register_provider(self, name: str, provider: ModelProvider) -> None:
        with self._lock:
            self._providers[name] = provider
            if self._active_provider is None:
                self._active_provider = name
    
    def get_provider(self, name: Optional[str] = None) -> Optional[ModelProvider]:
        with self._lock:
            if name is None:
                name = self._active_provider
            return self._providers.get(name)
    
    def get_config(self, name: str) -> Optional[ModelConfig]:
        return self._configs.get(name)
    
    def set_active(self, name: str) -> bool:
        with self._lock:
            if name in self._providers or name in self._configs:
                self._active_provider = name
                return True
            return False
    
    @property
    def active_name(self) -> Optional[str]:
        return self._active_provider
    
    def list_providers(self) -> List[str]:
        with self._lock:
            return list(self._providers.keys())
    
    def list_configs(self) -> List[str]:
        return list(self._configs.keys())
    
    def list_all(self) -> List[str]:
        with self._lock:
            all_names = set(self._providers.keys()) | set(self._configs.keys())
            return sorted(all_names)
    
    def get_or_create_provider(self, name: Optional[str] = None) -> ModelProvider:
        from .factory import create_provider
        
        if name is None:
            name = self._active_provider
        
        if name is None:
            raise ValueError("No model configured")
        
        with self._lock:
            if name in self._providers:
                provider = self._providers[name]
                if not provider._initialized:
                    provider.initialize()
                return provider
            
            if name in self._configs:
                config = self._configs[name]
                provider = create_provider(config)
                provider.initialize()
                self._providers[name] = provider
                return provider
        
        raise ValueError(f"Unknown model: {name}")
    
    def cleanup(self, name: Optional[str] = None) -> None:
        with self._lock:
            if name:
                if name in self._providers:
                    self._providers[name].cleanup()
                    del self._providers[name]
            else:
                for provider in self._providers.values():
                    provider.cleanup()
                self._providers.clear()
    
    def get_model_info(self, name: str) -> Dict:
        config = self._configs.get(name)
        provider = self._providers.get(name)
        
        if not config and not provider:
            return {}
        
        if config:
            return {
                "name": name,
                "provider_type": config.provider_type,
                "model_id": config.model_id,
                "capabilities": [c.name for c in config.capabilities],
                "initialized": provider._initialized if provider else False,
            }
        
        return {
            "name": name,
            "provider_type": provider.config.provider_type,
            "model_id": provider.config.model_id,
            "capabilities": [c.name for c in provider.capabilities],
            "initialized": provider._initialized,
        }
    
    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            if cls._instance is not None:
                for provider in cls._instance._providers.values():
                    try:
                        provider.cleanup()
                    except Exception:
                        pass
                cls._instance._providers.clear()
                cls._instance._configs.clear()
                cls._instance._active_provider = None
            cls._instance = None


def get_model_registry() -> ModelRegistry:
    return ModelRegistry()
