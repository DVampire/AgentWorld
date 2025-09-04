"""Prompt manager for managing agent prompt templates."""
from typing import Dict, Any, List
from langchain_core.messages import SystemMessage, HumanMessage

from src.prompts.system_prompt import SystemPrompt
from src.prompts.agent_message_prompt import AgentMessagePrompt
from src.prompts.templates import PROMPT_TEMPLATES

class PromptManager():
    """Manager for SystemPrompt and AgentMessagePrompt instances."""
    
    def __init__(self, prompt_name: str = "tool_calling", **kwargs):
        super().__init__(**kwargs)
        self.prompt_name = prompt_name
        
        self.system_prompt_name = f"{prompt_name}_system_prompt"
        self.agent_message_prompt_name = f"{prompt_name}_agent_message_prompt"
        
        self.system_prompt = SystemPrompt(prompt_name=self.system_prompt_name, **kwargs)
        self.agent_message_prompt = AgentMessagePrompt(prompt_name=self.agent_message_prompt_name, **kwargs)
    
    def get_system_message(self, input_variables: Dict[str, Any], **kwargs) -> SystemMessage:
        """Get a system message using SystemPrompt."""
        return self.system_prompt.get_message(input_variables, **kwargs)
    
    def get_agent_message(self, input_variables: Dict[str, Any], **kwargs) -> HumanMessage:
        """Get a system message using AgentMessagePrompt."""
        return self.agent_message_prompt.get_message(input_variables, **kwargs)
    
    def list_prompts(self) -> List[str]:
        """List all available prompts."""
        return list(PROMPT_TEMPLATES.keys())
    
    def get_system_prompt_template(self) -> str:
        """Get the system prompt template."""
        return self.system_prompt.get_prompt_template()
    
    def get_agent_message_prompt_template(self) -> str:
        """Get the agent message prompt template."""
        return self.agent_message_prompt.get_prompt_template()