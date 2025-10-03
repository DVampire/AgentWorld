"""Agent message prompt management for dynamic task-related prompts."""

from typing import Dict, List, Any
from langchain_core.messages import HumanMessage
from datetime import datetime
from jinja2 import Template

from src.logger import logger
from src.agents.prompts.templates import PROMPT_TEMPLATES

class AgentMessagePrompt:
    """Agent message prompt manager for dynamic task-related prompts (tool-calling agents)."""
    
    def __init__(
        self,
        prompt_name: str = "tool_calling_agent_message_prompt",
        max_actions_per_step: int = 10,
        current_step: int = 1,
        max_steps: int = 50,
        **kwargs
    ):
        self.prompt_name = prompt_name
        self.max_actions_per_step = max_actions_per_step
        self.current_step = current_step
        self.max_steps = max_steps
        
        self.template_str, self.input_vars = self._load_template()
        self.agent_message = None
    
    def get_message(self, input_variables: Dict[str, Any]) -> HumanMessage:
        """Get complete task state as a single message using template."""
        try:
            # Prepare variables for Jinja2 formatting
            variables = {
                key: value for key, value in input_variables.items() if key in self.input_vars
            }
            variables["step_info"] = self._build_step_info()
            
            # Use Jinja2 for template rendering
            jinja_template = Template(self.template_str)
            formatted_content = jinja_template.render(**variables)
            print(formatted_content)
            exit()
            
            return HumanMessage(content=formatted_content, cache=True)
            
        except Exception as e:
            logger.warning(f"Failed to render agent message template: {e}")
            return HumanMessage(content=self.template_str, cache=True)
        
    def get_prompt_template(self) -> str:
        """Get the prompt template."""
        return self.template_str
    
    def _build_step_info(self) -> str:
        """Build step info string."""
        step_info_description = f'Step {self.current_step} of {self.max_steps} max possible steps\n'
        time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        step_info_description += f'Current date and time: {time_str}'
        step_info_description += f'\nMax actions per step: {self.max_actions_per_step}'
        
        return step_info_description
    
    def _load_template(self) -> tuple[str, List[str]]:
        """Load the agent message template and its declared input variables.
        Returns (template_str, input_variables).
        """
        try:
            template = PROMPT_TEMPLATES[self.prompt_name]
            template_str = template.get('template', '')
            input_vars = template.get('input_variables', [])
            return template_str, input_vars
        except Exception as e:
            logger.warning(f"Failed to load agent message template: {e}")
            raise e