"""Prompt manager for managing agent prompt templates."""

import os
import importlib
from typing import Dict, List, Optional, Union
from langchain.prompts import PromptTemplate
from pathlib import Path

from src.utils import Singleton
from src.utils import assemble_project_path


class PromptManager(metaclass=Singleton):
    """Manager for all agent prompt templates."""
    
    def __init__(self):
        self.templates: Dict[str, PromptTemplate] = {}
        self.template_configs: Dict[str, Dict] = {}
        self._load_templates()
    
    def _load_templates(self):
        """Load all prompt templates from the templates directory."""
        templates_dir = Path(assemble_project_path("src/prompts/templates"))
        
        if not templates_dir.exists():
            print(f"Warning: Templates directory {templates_dir} does not exist")
            return
        
        # Load all Python files in the templates directory
        for template_file in templates_dir.glob("*.py"):
            if template_file.name.startswith("__"):
                continue
            
            try:
                # Import the template module
                module_name = f"src.prompts.templates.{template_file.stem}"
                module = importlib.import_module(module_name)
                
                # Look for PROMPT_TEMPLATES in the module
                if hasattr(module, "PROMPT_TEMPLATES"):
                    for template_name, template_config in module.PROMPT_TEMPLATES.items():
                        self._register_template(template_name, template_config)
                
                # Look for individual template variables
                for attr_name in dir(module):
                    if attr_name.endswith("_PROMPT") and not attr_name.startswith("_"):
                        template_config = getattr(module, attr_name)
                        template_name = attr_name.replace("_PROMPT", "").lower()
                        self._register_template(template_name, template_config)
                        
            except Exception as e:
                print(f"Error loading templates from {template_file}: {e}")
    
    def _register_template(self, name: str, config: Union[str, Dict]):
        """Register a prompt template."""
        if isinstance(config, str):
            # Simple string template
            template = PromptTemplate.from_template(config)
            self.templates[name] = template
            self.template_configs[name] = {"template": config, "type": "simple"}
        elif isinstance(config, dict):
            # Complex template configuration
            template_str = config.get("template", "")
            input_variables = config.get("input_variables", [])
            partial_variables = config.get("partial_variables", {})
            
            template = PromptTemplate(
                template=template_str,
                input_variables=input_variables,
                partial_variables=partial_variables
            )
            
            self.templates[name] = template
            self.template_configs[name] = config
        else:
            print(f"Warning: Invalid template configuration for {name}")
    
    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """Get a prompt template by name."""
        return self.templates.get(name)
    
    def get_template_config(self, name: str) -> Optional[Dict]:
        """Get template configuration by name."""
        return self.template_configs.get(name)
    
    def list_templates(self) -> List[str]:
        """List all available template names."""
        return list(self.templates.keys())
    
    def add_template(self, name: str, template: Union[str, Dict, PromptTemplate]):
        """Add a new prompt template."""
        if isinstance(template, PromptTemplate):
            self.templates[name] = template
            self.template_configs[name] = {"type": "custom"}
        else:
            self._register_template(name, template)
    
    def remove_template(self, name: str) -> bool:
        """Remove a prompt template."""
        if name in self.templates:
            del self.templates[name]
            del self.template_configs[name]
            return True
        return False
    
    def update_template(self, name: str, template: Union[str, Dict, PromptTemplate]):
        """Update an existing prompt template."""
        if name in self.templates:
            self.add_template(name, template)
            return True
        return False
    
    def get_template_variables(self, name: str) -> List[str]:
        """Get input variables for a template."""
        template = self.get_template(name)
        if template:
            return template.input_variables
        return []
    
    def validate_template(self, name: str, variables: Dict) -> bool:
        """Validate that all required variables are provided."""
        template = self.get_template(name)
        if not template:
            return False
        
        required_vars = set(template.input_variables)
        provided_vars = set(variables.keys())
        
        return required_vars.issubset(provided_vars)
    
    def format_template(self, name: str, **kwargs) -> str:
        """Format a template with provided variables."""
        template = self.get_template(name)
        if not template:
            raise ValueError(f"Template '{name}' not found")
        
        if not self.validate_template(name, kwargs):
            missing_vars = set(template.input_variables) - set(kwargs.keys())
            raise ValueError(f"Missing required variables: {missing_vars}")
        
        return template.format(**kwargs)
    
    def reload_templates(self):
        """Reload all templates from files."""
        self.templates.clear()
        self.template_configs.clear()
        self._load_templates()
    
    def export_templates(self, file_path: str):
        """Export all templates to a file."""
        import json
        
        export_data = {
            "templates": {},
            "configs": self.template_configs
        }
        
        for name, template in self.templates.items():
            export_data["templates"][name] = {
                "template": template.template,
                "input_variables": template.input_variables,
                "partial_variables": template.partial_variables
            }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    def import_templates(self, file_path: str):
        """Import templates from a file."""
        import json
        
        with open(file_path, 'r', encoding='utf-8') as f:
            import_data = json.load(f)
        
        for name, template_data in import_data.get("templates", {}).items():
            template = PromptTemplate(
                template=template_data["template"],
                input_variables=template_data["input_variables"],
                partial_variables=template_data.get("partial_variables", {})
            )
            self.templates[name] = template
        
        self.template_configs.update(import_data.get("configs", {}))
        
prompt_manager = PromptManager()
