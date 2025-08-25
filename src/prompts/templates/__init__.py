from .tool_calling import PROMPT_TEMPLATES as TOOL_CALLING_PROMPT_TEMPLATES

# This module contains all prompt templates for different agent types
PROMPT_TEMPLATES = {
    **TOOL_CALLING_PROMPT_TEMPLATES,
}