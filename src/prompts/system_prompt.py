"""System prompt management for agents."""

import importlib
from typing import Dict, Any, List
from langchain_core.messages import SystemMessage
from jinja2 import Template

from src.logger import logger
from src.prompts.templates import PROMPT_TEMPLATES


class SystemPrompt:
    """System prompt manager for tool-calling agents (static constitution).

    Parameters
    - prompt_name: selects a template entry from PROMPT_TEMPLATES if present (default: "tool_calling_system_prompt")
    - max_actions_per_step: number used to format {{ max_actions }} in the template
    """

    def __init__(
        self,
        prompt_name: str = "tool_calling_system_prompt",
        max_actions_per_step: int = 10,
        **kwargs
    ):
        self.prompt_name = prompt_name
        self.max_actions_per_step = max_actions_per_step
        
        self.template_str, self.input_vars = self._load_prompt_template()
        self.system_message = None

    def _load_prompt_template(self) -> tuple[str, List[str]]:
        """Load the tool-calling system prompt template and its declared input variables.
        Returns (template_str, input_variables).
        """
        try:
            template = PROMPT_TEMPLATES[self.prompt_name]
            template_str = template.get('template', '')
            input_vars = template.get('input_variables', [])
            return template_str, input_vars
        except Exception as e:
            logger.warning(f"Failed to load system prompt template: {e}")
            raise e
        
    def get_prompt_template(self) -> str:
        """Get the prompt template."""
        return self.template_str

    def get_message(self) -> SystemMessage:
        """Get the system prompt for the agent."""
        if self.system_message:
            return self.system_message
        try:
            variables: Dict[str, Any] = {}
            if "max_actions" in self.input_vars:
                variables["max_actions"] = self.max_actions_per_step
            jinja_template = Template(self.template_str)
            prompt = jinja_template.render(**variables) if variables else self.template_str
            self.system_message = SystemMessage(content=prompt, cache=True)
        except Exception as e:
            logger.warning(f"Failed to render system prompt template: {e}")
            self.system_message = SystemMessage(content=self.template_str, cache=True)
        return self.system_message
