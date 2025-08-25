#!/usr/bin/env python3
"""Test the new prompt management structure."""

import sys
import os
from pathlib import Path

# Add the project root to the path
root = str(Path(__file__).resolve().parents[1])
sys.path.append(root)

from src.prompts.system_prompt import SystemPrompt
from src.prompts.agent_message_prompt import AgentMessagePrompt
from src.prompts.prompt_manager import PromptManager

def test_system_prompt():
    """Test SystemPrompt class."""
    print("Testing SystemPrompt...")
    
    # Test default system prompt
    system_prompt = SystemPrompt(prompt_name="tool_calling_system_prompt")
    print(f"System prompt template: {system_prompt.get_prompt_template()[:200]}...")
    
    # Test system message
    system_message = system_prompt.get_message()
    print(f"System message content: {system_message.content[:200]}...")

def test_agent_message_prompt():
    """Test AgentMessagePrompt class."""
    print("\nTesting AgentMessagePrompt...")
    
    # Test agent message prompt with sample data
    agent_message_prompt = AgentMessagePrompt(prompt_name="tool_calling_agent_message_prompt")
    
    # Test user message generation
    agent_message = agent_message_prompt.get_message(
        input_variables={
            "task": "Create a Python script to calculate fibonacci numbers",
            "workdir": "/tmp",
            "agent_history": "Previous step: Analyzed requirements for fibonacci calculation",
            "file_system": "Working directory: /home/user/project\nFiles: main.py, utils.py",
            "todo_contents": "1. Implement fibonacci function\n2. Add input validation\n3. Test the function",
        }
    )
    print(f"Agent message content:\n{agent_message.content}")
    
def test_prompt_manager():
    """Test PromptManager class."""
    print("\nTesting PromptManager...")
    
    prompt_manager = PromptManager(prompt_name="tool_calling")
    
    system_message = prompt_manager.get_system_message()
    print(f"System message content:\n{system_message.content[:200]}...")
    
    agent_message = prompt_manager.get_agent_message(
        input_variables={
            "task": "Create a Python script to calculate fibonacci numbers",
            "workdir": "/tmp",
            "agent_history": "Previous step: Analyzed requirements for fibonacci calculation",
            "file_system": "Working directory: /home/user/project\nFiles: main.py, utils.py",
            "todo_contents": "1. Implement fibonacci function\n2. Add input validation\n3. Test the function",
        }
    )
    print(f"Agent message content:\n{agent_message.content[:200]}...")

if __name__ == "__main__":
    test_system_prompt()
    test_agent_message_prompt()
    test_prompt_manager()
    print("\nAll tests completed!")
