"""Model manager for managing language models."""

import os
import json
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

from src.utils import Singleton
from .base_model import BaseModel
from .openai_model import OpenAIAsyncModel
from .anthropic_model import AnthropicAsyncModel


class ModelManager(metaclass=Singleton):
    """Manager for all language models."""
    
    def __init__(self):
        self.models: Dict[str, BaseModel] = {}
        self.model_configs: Dict[str, Dict] = {}
        self._load_default_models()
    
    def _load_default_models(self):
        """Load default model configurations."""
        # Default OpenAI models
        self.add_model("gpt-4", OpenAIAsyncModel.create_gpt4("gpt-4"))
        self.add_model("gpt-3.5-turbo", OpenAIAsyncModel.create_gpt35_turbo("gpt-3.5-turbo"))
        self.add_model("gpt-4-turbo", OpenAIAsyncModel.create_gpt4_turbo("gpt-4-turbo"))
        
        # Default Anthropic models
        self.add_model("claude-3-sonnet", AnthropicAsyncModel.create_claude3_sonnet("claude-3-sonnet"))
        self.add_model("claude-3-opus", AnthropicAsyncModel.create_claude3_opus("claude-3-opus"))
        self.add_model("claude-3-haiku", AnthropicAsyncModel.create_claude3_haiku("claude-3-haiku"))
    
    def add_model(self, name: str, model: BaseModel):
        """Add a model to the manager."""
        self.models[name] = model
        self.model_configs[name] = model.get_config()
    
    def remove_model(self, name: str) -> bool:
        """Remove a model from the manager."""
        if name in self.models:
            del self.models[name]
            del self.model_configs[name]
            return True
        return False
    
    def get_model(self, name: str) -> Optional[BaseModel]:
        """Get a model by name."""
        return self.models.get(name)
    
    def get_model_config(self, name: str) -> Optional[Dict]:
        """Get model configuration by name."""
        return self.model_configs.get(name)
    
    def list_models(self) -> List[str]:
        """List all available model names."""
        return list(self.models.keys())
    
    def list_models_by_type(self, model_type: str) -> List[str]:
        """List models by type (openai, anthropic, etc.)."""
        return [name for name, model in self.models.items() if model.model_type == model_type]
    
    def list_openai_models(self) -> List[str]:
        """List all OpenAI models."""
        return self.list_models_by_type("openai")
    
    def list_anthropic_models(self) -> List[str]:
        """List all Anthropic models."""
        return self.list_models_by_type("anthropic")
    
    def create_openai_model(self, name: str, model_name: str = "gpt-4", **kwargs) -> OpenAIAsyncModel:
        """Create and add an OpenAI model."""
        model = OpenAIAsyncModel(name, model_name, **kwargs)
        self.add_model(name, model)
        return model
    
    def create_anthropic_model(self, name: str, model_name: str = "claude-3-sonnet-20240229", **kwargs) -> AnthropicAsyncModel:
        """Create and add an Anthropic model."""
        model = AnthropicAsyncModel(name, model_name, **kwargs)
        self.add_model(name, model)
        return model
    
    def update_model_config(self, name: str, new_config: Dict[str, Any]):
        """Update model configuration."""
        model = self.get_model(name)
        if model:
            model.update_config(new_config)
            self.model_configs[name] = model.get_config()
        else:
            raise ValueError(f"Model '{name}' not found")
    
    def get_model_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a model."""
        model = self.get_model(name)
        if model:
            return model.get_model_info()
        return None
    
    def validate_model(self, name: str) -> bool:
        """Validate that a model exists and is properly configured."""
        model = self.get_model(name)
        if not model:
            return False
        
        # Check if model has required API keys
        if model.model_type == "openai":
            api_key = model.config.get("api_key")
            if not api_key:
                print(f"Warning: OpenAI model '{name}' missing API key")
                return False
        elif model.model_type == "anthropic":
            api_key = model.config.get("api_key")
            if not api_key:
                print(f"Warning: Anthropic model '{name}' missing API key")
                return False
        
        return True
    
    def get_available_models(self) -> Dict[str, List[str]]:
        """Get all available models grouped by type."""
        return {
            "openai": self.list_openai_models(),
            "anthropic": self.list_anthropic_models(),
            "all": self.list_models()
        }
    
    def export_models(self, file_path: str):
        """Export all model configurations to a file."""
        export_data = {
            "models": {},
            "configs": self.model_configs
        }
        
        for name, model in self.models.items():
            export_data["models"][name] = {
                "type": model.model_type,
                "config": model.get_config()
            }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    def import_models(self, file_path: str):
        """Import models from a file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            import_data = json.load(f)
        
        for name, model_data in import_data.get("models", {}).items():
            model_type = model_data["type"]
            config = model_data["config"]
            
            if model_type == "openai":
                model = OpenAIAsyncModel(name, **config)
            elif model_type == "anthropic":
                model = AnthropicAsyncModel(name, **config)
            else:
                print(f"Warning: Unknown model type '{model_type}' for model '{name}'")
                continue
            
            self.add_model(name, model)
        
        self.model_configs.update(import_data.get("configs", {}))
    
    def reload_models(self):
        """Reload all models from configurations."""
        for name, config in self.model_configs.items():
            model_type = config.get("type", "openai")
            
            if model_type == "openai":
                model = OpenAIAsyncModel(name, **config)
            elif model_type == "anthropic":
                model = AnthropicAsyncModel(name, **config)
            else:
                continue
            
            self.models[name] = model
    
    def get_model_statistics(self) -> Dict[str, Any]:
        """Get statistics about all models."""
        total_models = len(self.models)
        openai_models = len(self.list_openai_models())
        anthropic_models = len(self.list_anthropic_models())
        
        valid_models = sum(1 for name in self.models.keys() if self.validate_model(name))
        
        return {
            "total_models": total_models,
            "openai_models": openai_models,
            "anthropic_models": anthropic_models,
            "valid_models": valid_models,
            "invalid_models": total_models - valid_models
        }
