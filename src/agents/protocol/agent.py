"""Base agent class for multi-agent system."""
from typing import List, Optional, Type, Dict, Any, Union
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field, ConfigDict
import pandas as pd

from src.logger import logger
from src.infrastructures.models import model_manager
from src.agents.prompts import PromptManager
from src.infrastructures.memory import MemoryManager

def format_actions(actions: List[BaseModel]) -> str:
    """Format actions as a Markdown table using pandas."""
    rows = []
    for action in actions:
        if isinstance(action.args, dict):
            args_str = ", ".join(f"{k}={v}" for k, v in action.args.items())
        else:
            args_str = str(action.args)

        rows.append({
            "Action": action.name,
            "Args": args_str,
            "Output": action.output if action.output is not None else None
        })
    
    df = pd.DataFrame(rows)
    
    if df["Output"].isna().all():
        df = df.drop(columns=["Output"])
    else:
        df["Output"] = df["Output"].fillna("None")
    
    return df.to_markdown(index=True)

class ThinkOutputBuilder:
    def __init__(self):
        self.schemas: Dict[str, type[BaseModel]] = {}

    def register(self, schema: Dict[str, type[BaseModel]]):
        """Register new args schema"""
        self.schemas.update(schema)
        return self  # Support chaining

    def build(self):
        """Generate Action and ThinkOutput models"""

        # -------- Dynamically generate Action --------
        schemas = self.schemas
        ActionArgs = Union[tuple(schemas.values())]

        class Action(BaseModel):
            name: str = Field(description="The name of the action.")
            args: ActionArgs = Field(description="The arguments of the action.")
            output: Optional[str] = Field(default=None, description="The output of the action.")
            
            def __str__(self):
                return f"Action: {self.name}\nArgs: {self.args}\nOutput: {self.output}\n"
            
            def __repr__(self):
                return self.__str__()

        # -------- Dynamically generate ThinkOutput --------
        class ThinkOutput(BaseModel):
            thinking: str = Field(description="A structured <think>-style reasoning block.")
            evaluation_previous_goal: str = Field(description="One-sentence analysis of your last action.")
            memory: str = Field(description="1-3 sentences of specific memory.")
            next_goal: str = Field(description="State the next immediate goals and actions.")
            action: List[Action] = Field(
                description='[{"name": "action_name", "args": {...}}, ...]'
            )

            def __str__(self):
                return (
                    f"Thinking: {self.thinking}\n"
                    f"Evaluation of Previous Goal: {self.evaluation_previous_goal}\n"
                    f"Memory: {self.memory}\n"
                    f"Next Goal: {self.next_goal}\n"
                    f"Action:\n{format_actions(self.action)}\n"
                )
            
            def __repr__(self):
                return self.__str__()

        return ThinkOutput

class BaseAgent(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    name: str = Field(description="The name of the agent.")
    type: str = Field(description="The type of the agent.")
    description: str = Field(description="The description of the agent.")
    args_schema: Type[BaseModel] = Field(description="The args schema of the agent.")
    metadata: Dict[str, Any] = Field(description="The metadata of the agent.")
    
    def __init__(
        self,
        workdir: str,
        model_name: Optional[str] = None,
        prompt_name: Optional[str] = None,
        max_steps: int = 20,
        review_steps: int = 5,
        log_max_length: int = 1000,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.workdir = workdir
        logger.info(f"| 📁 Agent working directory: {self.workdir}")
        
        self.prompt_manager = PromptManager(prompt_name=prompt_name)
        self.memory_manager = MemoryManager()
        
        # Setup model
        self.model = self._setup_model(model_name)
        
        # Setup steps
        self.max_steps = max_steps if max_steps>0 else int(1e8)
        self.review_steps = review_steps
        self.step_number = 0
        self.log_max_length = log_max_length

    def _setup_model(self, model_name: Optional[str]):
        """Setup the language model."""
        if model_name:
            # Get model from ModelManager
            model = model_manager.get(model_name)
            if model:
                return model
            else:
                logger.warning(f"Warning: Model '{model_name}' not found in model_manager")
        
        # Fallback to default model
        default_model = model_manager.get("gpt-4.1")
        if default_model:
            return default_model
        else:
            raise RuntimeError("No model available")
    
    def __str__(self):
        return f"Agent(name={self.name}, model={self.model_name}, prompt_name={self.prompt_name})"
    
    def __repr__(self):
        return self.__str__()
        
    async def get_messages(self, task: str) -> List[BaseMessage]:
        raise NotImplementedError("Get messages method is not implemented by the child class")

    async def ainvoke(self,  task: str, files: Optional[List[str]] = None):
        """Run the agent. This method should be implemented by the child classes."""
        raise NotImplementedError("Run method is not implemented by the child class")
    
    def invoke(self,  task: str, files: Optional[List[str]] = None):
        """Run the agent. This method should be implemented by the child classes."""
        raise NotImplementedError("Run method is not implemented by the child class")
