"""Tracer module for recording agent execution records."""

import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, Field


class Record(BaseModel):
    """Record model for agent execution records."""
    
    id: Optional[int] = Field(default=None, description="Unique identifier for the record")
    observation: Optional[Any] = Field(default=None, description="Observation data for this execution step")
    action: Optional[Any] = Field(default=None, description="Action taken in this execution step")
    timestamp: Optional[str] = Field(default=None, description="Timestamp of the record in ISO format")

class Tracer:
    """Tracer class for recording agent execution records.
    
    This class maintains a list of execution records, where each record
    is a Record model containing observation, action, id, and timestamp information.
    """
    
    def __init__(self):
        """Initialize the Tracer with an empty records list."""
        self.records: List[Record] = []
        self._next_id: int = 1
    
    def add_record(
        self,
        observation: Any,
        action: Any,
        timestamp: Optional[datetime] = None
    ) -> None:
        """Add a new execution record.
        
        Args:
            observation: The observation data for this execution step
            action: The action taken in this execution step
            timestamp: Optional timestamp for the record. If None, uses current time.
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        record = Record(
            id=self._next_id,
            observation=observation,
            action=action,
            timestamp=timestamp.isoformat()
        )
        
        self._next_id += 1
        self.records.append(record)
    
    def get_records(self) -> List[Record]:
        """Get all execution records.
        
        Returns:
            A list of all execution records.
        """
        return self.records.copy()
    
    def get_record(self, index: int) -> Optional[Record]:
        """Get a specific execution record by index.
        
        Args:
            index: The index of the record to retrieve.
            
        Returns:
            The record at the specified index, or None if index is out of range.
        """
        if 0 <= index < len(self.records):
            return self.records[index]
        return None
    
    def get_record_by_id(self, record_id: int) -> Optional[Record]:
        """Get a specific execution record by ID.
        
        Args:
            record_id: The ID of the record to retrieve.
            
        Returns:
            The record with the specified ID, or None if not found.
        """
        for record in self.records:
            if record.id == record_id:
                return record
        return None
    
    def clear(self) -> None:
        """Clear all execution records."""
        self.records.clear()
        self._next_id = 1
    
    def save_to_json(self, file_path: str) -> None:
        """Save all records to a JSON file.
        
        Args:
            file_path: Path to the JSON file where records will be saved.
            
        Raises:
            IOError: If the file cannot be written.
        """
        file_path = Path(file_path)
        
        # Create directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert records to JSON-serializable format
        json_records = []
        for record in self.records:
            json_record = {
                "id": record.id,
                "observation": self._serialize_for_json(record.observation),
                "action": self._serialize_for_json(record.action),
                "timestamp": record.timestamp
            }
            json_records.append(json_record)
        
        # Write to JSON file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(json_records, f, indent=4, ensure_ascii=False)
    
    def load_from_json(self, file_path: str) -> None:
        """Load records from a JSON file.
        
        Args:
            file_path: Path to the JSON file to load records from.
            
        Raises:
            FileNotFoundError: If the file does not exist.
            json.JSONDecodeError: If the file contains invalid JSON.
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"JSON file not found: {file_path}")
        
        # Read from JSON file
        with open(file_path, 'r', encoding='utf-8') as f:
            json_records = json.load(f)
        
        # Convert JSON records back to internal format
        self.records = []
        max_id = 0
        for json_record in json_records:
            record_id = json_record.get("id")
            if record_id is not None:
                max_id = max(max_id, record_id)
            
            record = Record(
                id=record_id,
                observation=json_record.get("observation"),
                action=json_record.get("action"),
                timestamp=json_record.get("timestamp")
            )
            self.records.append(record)
        
        # Set next_id to continue from the maximum id found
        self._next_id = max_id + 1 if max_id > 0 else 1
    
    def _serialize_for_json(self, obj: Any) -> Any:
        """Serialize an object for JSON encoding.
        
        Args:
            obj: Object to serialize.
            
        Returns:
            JSON-serializable representation of the object.
        """
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        elif isinstance(obj, dict):
            return {k: self._serialize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._serialize_for_json(item) for item in obj]
        elif isinstance(obj, datetime):
            return obj.isoformat()
        else:
            # For other types, convert to string
            return str(obj)
    
    def __len__(self) -> int:
        """Return the number of records."""
        return len(self.records)
    
    def __repr__(self) -> str:
        """Return a string representation of the Tracer."""
        return f"Tracer(records={len(self.records)})"
    
    def __str__(self) -> str:
        """Return a string representation of the Tracer."""
        return self.__repr__()

