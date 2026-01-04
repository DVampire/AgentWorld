from typing import Dict, Any, Optional, List, Tuple, Type
from pydantic import BaseModel, Field, PrivateAttr, ConfigDict
import inflection

from src.logger import logger
from src.dynamic import dynamic_manager

class Benchmark(BaseModel):
    """Base class for all benchmark systems"""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    name: str = Field(default="", description="The name of the benchmark")
    description: str = Field(default="", description="The description of the benchmark")
    
    # Dataset-related fields
    split: str = Field(default="test", description="Dataset split")
    subset: Optional[str] = Field(default=None, description="Subset name")
    path: str = Field(default="", description="Dataset path")
    
    # Private attributes for state management
    _dataset_instance: Any = PrivateAttr(default=None)
    _ground_truth_map: Dict[str, str] = PrivateAttr(default_factory=dict)
    _data_records: List[Dict] = PrivateAttr(default_factory=list)
    _current_index: int = PrivateAttr(default=0)
    _results: List[Dict] = PrivateAttr(default_factory=list)

    def __init__(self, **kwargs):
        """Initialize benchmark system and instantiate data."""
        super().__init__(**kwargs)
        # Auto-set name from class name if not provided
        if not self.name:
            self.name = inflection.underscore(self.__class__.__name__)
        # Auto-set description from docstring if not provided
        if not self.description and self.__class__.__doc__:
            self.description = self.__class__.__doc__.strip().split('\n')[0]
            
        # Each benchmark should instantiate its own data during init
        self._dataset_instance = self._instantiate_dataset()
        
        # Automatically load data records and ground truth if dataset is available
        if self._dataset_instance is not None and hasattr(self._dataset_instance, 'data'):
            full_df = self._dataset_instance.data
            if "task_id" in full_df.columns and "true_answer" in full_df.columns:
                self._ground_truth_map = dict(zip(
                    full_df["task_id"].astype(str), 
                    full_df["true_answer"].astype(str)
                ))
            
            # Cache data for step() iteration
            self._data_records = full_df.to_dict(orient="records")
            
            # Initialize state
            self.reset()
            logger.info(f"| [{self.name}] ✅ Data instantiated and loaded. Size: {len(self._data_records)}")

    def _instantiate_dataset(self) -> Any:
        """Instantiate the dataset. To be implemented by subclasses."""
        return None

    def get_task_description(self) -> str:
        raise NotImplementedError

    # ================= State and Control Functions =================

    def reset(self) -> Optional[Dict[str, Any]]:
        """
        Reset evaluation progress and statistics. Returns the first task.
        """
        self._current_index = 0
        self._results = []
        logger.info(f"| [{self.name}] ✅ Progress reset. Ready to start.")
        return self.step()

    def step(self) -> Optional[Dict[str, Any]]:
        """Get the next task to be tested."""
        if self._current_index >= len(self._data_records):
            return None
        
        # Get current record
        record = self._data_records[self._current_index]
        self._current_index += 1
        
        # Construct task object
        task_id = str(record.get("task_id") or record.get("id",""))
        task = {
            "task_id": task_id,
            "input": record.get("question") or record.get("prompt") or "",
            "system_prompt": self.get_task_description(),
            "ground_truth": record.get("true_answer") or record.get("answer") or self._ground_truth_map.get(task_id),
            **{k: v for k, v in record.items() if k not in ["true_answer", "answer", "task_id", "id"]}
        }
        return task

    @property
    def progress(self) -> Tuple[int, int]:
        """Return (current index, total count)."""
        return self._current_index, len(self._data_records)

    # ================= Evaluation and Statistics Functions =================

    async def eval_task(self, prediction: str, ground_truth: Optional[str] = None, task_id: Optional[str] = None, **kwargs) -> float:
        """Public interface for single task evaluation."""
        if ground_truth is None and task_id is not None:
            ground_truth = self._ground_truth_map.get(str(task_id))
        
        if ground_truth is None:
            logger.error(f"| [{self.name}] ❌ Task ID '{task_id}' not found and no ground truth provided.")
            score = 0.0
            ground_truth = "N/A"
        else:
            score = await self._eval_logic(prediction, ground_truth, task_id=str(task_id) if task_id else None, **kwargs)
            
        # Record state
        result_entry = {
            "task_id": task_id,
            "prediction": prediction,
            "ground_truth": ground_truth,
            "score": score,
            "metadata": kwargs
        }
        self._results.append(result_entry)
        
        return score

    async def _eval_logic(self, prediction: str, ground_truth: str, **kwargs) -> float:
        raise NotImplementedError

    def get_stats(self) -> Dict[str, Any]:
        """Calculate current overall statistics."""
        total_attempted = len(self._results)
        if total_attempted == 0:
            return {"accuracy": 0.0, "total": 0, "correct": 0}
        
        correct_count = sum(1 for r in self._results if r["score"] >= 1.0)
        accuracy = correct_count / total_attempted
        
        return {
            "accuracy": accuracy,
            "total_attempted": total_attempted,
            "correct_count": correct_count,
            "total_dataset_size": len(self._data_records),
            "progress_percent": f"{(total_attempted / len(self._data_records))*100:.1f}%" if len(self._data_records) > 0 else "0%"
        }

    def get_results(self) -> List[Dict]:
        """Get detailed results list."""
        return self._results


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
