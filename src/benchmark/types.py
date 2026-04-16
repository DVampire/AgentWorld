import os
from typing import Dict, Any, Optional, List, Tuple, Type
from pydantic import BaseModel, Field, PrivateAttr, ConfigDict
import inflection

from src.logger import logger
from src.dynamic import dynamic_manager
from src.config import config
from src.utils import assemble_project_path

class Task(BaseModel):
    """Data model for a single benchmark task"""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    # Input
    task_id: str = Field(description="Unique identifier for the task")
    input: str = Field(description="The input prompt/question for the task")
    system_prompt: Optional[str] = Field(default=None, description="The system prompt for the task")
    ground_truth: Optional[Any] = Field(default=None, description="The expected correct answer")
    
    # Output
    reasoning: Optional[str] = Field(default=None, description="The reasoning process")
    result: Optional[Any] = Field(default=None, description="The final answer")
    time: Optional[float] = Field(default=0.0, description="The time taken to complete the task in seconds")
    score: Optional[float] = Field(default=0.0, description="The score of the task")
    
    extra: Optional[Dict[str, Any]] = Field(default=None, description="Additional task-specific metadata")


class Stats(BaseModel):
    """Data model for benchmark statistics"""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    accuracy: float = Field(default=0.0, description="Overall accuracy score")
    total: int = Field(default=0, description="Total number of tasks")
    correct: int = Field(default=0, description="Number of correct tasks")
    wrong: int = Field(default=0, description="Number of wrong tasks")
    times: Dict[str, float] = Field(default_factory=dict, description="Time taken for each task (task_id -> seconds)")
    average_time: float = Field(default=0.0, description="Average time per task in seconds")
    extra: Dict[str, Any] = Field(default_factory=dict, description="Additional statistics or information")

class Benchmark(BaseModel):
    """Base class for all benchmark systems"""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str = Field(default="", description="The name of the benchmark")
    description: str = Field(default="", description="The description of the benchmark")

    # Dataset-related fields
    split: str = Field(default="test", description="Dataset split")
    subset: Optional[str] = Field(default=None, description="Subset name")
    path: str = Field(default="", description="Dataset path")
    base_dir: str = Field(default="", description="Base directory for storing benchmark outputs and results")
    start: Optional[int] = Field(default=None, description="Start index for slicing the dataset (inclusive). None means from the beginning.")
    end: Optional[int] = Field(default=None, description="End index for slicing the dataset (exclusive). None means to the last item.")

    def __init__(self, base_dir: Optional[str] = None, start: Optional[int] = None, end: Optional[int] = None, **kwargs):
        """Initialize benchmark system."""
        super().__init__(start=start, end=end, **kwargs)
        # Auto-set name from class name if not provided
        if not self.name:
            self.name = inflection.underscore(self.__class__.__name__)
        # Auto-set description from docstring if not provided
        if not self.description and self.__class__.__doc__:
            self.description = self.__class__.__doc__.strip().split('\n')[0]
        # Set base_dir
        if base_dir is not None:
            self.base_dir = assemble_project_path(base_dir)
        else:
            self.base_dir = assemble_project_path(
                os.path.join(config.workdir, "benchmark", self.name)
            )

    def _apply_slice(self, records: list) -> list:
        """Slice records according to start/end fields."""
        if self.start is None and self.end is None:
            return records
        return records[self.start:self.end]

    async def initialize(self) -> Any:
        """Instantiate the dataset. To be implemented by subclasses."""
        raise NotImplementedError

    async def reset(self) -> Optional[Task]:
        """
        Reset evaluation progress and statistics. Returns the first task.
        """
        raise NotImplementedError

    async def step(self) -> Optional[Task]:
        """Get the next task to be tested."""
        raise NotImplementedError

    async def eval(self, task: Task) -> Optional[Task]:
        """Public interface for single task evaluation."""
        raise NotImplementedError

    async def stats(self) -> Optional[Stats]:
        """Calculate current overall statistics."""
        raise NotImplementedError
    
    async def save_result(self, task: Task) -> None:
        """Save the result for a completed task."""
        raise NotImplementedError
    
    async def cleanup(self):
        """Cleanup benchmark resources."""
        pass
    

class BenchmarkConfig(BaseModel):
    """Benchmark configuration for registration"""
    name: str = Field(description="The name of the benchmark")
    description: str = Field(description="The description of the benchmark")
    version: str = Field(default="1.0.0", description="Version of the benchmark")
    
    cls: Optional[Type[Benchmark]] = Field(default=None, description="The class of the benchmark")
    instance: Optional[Any] = Field(default=None, description="The instance of the benchmark")
    config: Optional[Dict[str, Any]] = Field(default_factory=dict, description="The initialization configuration")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="The metadata")
    code: Optional[str] = Field(default=None, description="Source code")
    
    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Dump the model to a dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "cls": dynamic_manager.get_class_string(self.cls) if self.cls else None,
            "config": self.config,
            "instance": None,  # Don't serialize instance
            "metadata": self.metadata,
            "code": self.code,
        }
    
    @classmethod
    def model_validate(cls, data: Dict[str, Any]) -> 'BenchmarkConfig':
        """Validate the model from a dictionary."""
        name = data.get("name")
        description = data.get("description")
        version = data.get("version", "1.0.0")
        
        cls_ = None
        code = data.get("code")
        if code:
            class_name = dynamic_manager.extract_class_name_from_code(code)
            if class_name:
                try:
                    cls_ = dynamic_manager.load_class(
                        code, 
                        class_name=class_name,
                        base_class=Benchmark,
                        context="benchmark"
                    )
                except Exception:
                    cls_ = None
        
        config = data.get("config", {})
        instance = data.get("instance", None)
        metadata = data.get("metadata", {})
        
        return cls(
            name=name,
            description=description,
            version=version,
            cls=cls_,
            config=config,
            instance=instance,
            metadata=metadata,
            code=code,
        )
