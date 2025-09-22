"""System prompt management for agents."""

from typing import Dict, Any, List
from langchain_core.messages import SystemMessage
from jinja2 import Template

from src.logger import logger
from src.agents.prompts.templates import PROMPT_TEMPLATES

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

    def get_message(self, input_variables: Dict[str, Any]) -> SystemMessage:
        """Get the system prompt for the agent."""
        if self.system_message:
            return self.system_message
        try:
            # Prepare variables for Jinja2 formatting
            variables = {
                key: value for key, value in input_variables.items() if key in self.input_vars
            }
            if "max_actions" not in variables:
                variables["max_actions"] = self.max_actions_per_step
                
            # Use Jinja2 for template rendering
            jinja_template = Template(self.template_str)
            prompt = jinja_template.render(**variables)
            
            self.system_message = SystemMessage(content=prompt, cache=True)
        except Exception as e:
            logger.warning(f"Failed to render system prompt template: {e}")
            self.system_message = SystemMessage(content=self.template_str, cache=True)
        return self.system_message
