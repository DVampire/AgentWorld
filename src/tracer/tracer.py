"""Tracer module for recording agent execution records."""

import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, Field


class Record(BaseModel):
    """Record model for agent execution records."""
    
    id: Optional[int] = Field(default=None, description="Unique identifier for the record")
    session_id: Optional[str] = Field(default=None, description="Session ID for this record")
    task_id: Optional[str] = Field(default=None, description="Task ID for this record")
    observation: Optional[Any] = Field(default=None, description="Observation data for this execution step")
    action: Optional[Any] = Field(default=None, description="Action taken in this execution step")
    timestamp: Optional[str] = Field(default=None, description="Timestamp of the record in ISO format")

class Tracer:
    """Tracer class for recording agent execution records.
    
    This class maintains execution records organized by session_id, where each record
    is a Record model containing observation, action, id, session_id, task_id, and timestamp information.
    """
    
    def __init__(self):
        """Initialize the Tracer with session-based record management."""
        # Dictionary mapping session_id to list of records
        self.session_records: Dict[str, List[Record]] = {}
        self.current_session_id: Optional[str] = None
        self._next_id: int = 1
    
    def add_record(
        self,
        observation: Any,
        action: Any,
        session_id: Optional[str] = None,
        task_id: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ) -> None:
        """Add a new execution record.
        
        Args:
            observation: The observation data for this execution step
            action: The action taken in this execution step
            session_id: Optional session ID for this record. If None, records are stored without session grouping.
            task_id: Optional task ID for this record.
            timestamp: Optional timestamp for the record. If None, uses current time.
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        record = Record(
            id=self._next_id,
            session_id=session_id,
            task_id=task_id,
            observation=observation,
            action=action,
            timestamp=timestamp.isoformat()
        )
        
        self._next_id += 1
        
        # Update current session ID
        if session_id:
            self.current_session_id = session_id
        
        # Add to session-specific records
        if session_id:
            if session_id not in self.session_records:
                self.session_records[session_id] = []
            self.session_records[session_id].append(record)
        else:
            # If no session_id, store in a special "default" session
            default_session = "_no_session"
            if default_session not in self.session_records:
                self.session_records[default_session] = []
            self.session_records[default_session].append(record)
    
    def get_records(self, session_id: Optional[str] = None) -> List[Record]:
        """Get execution records.
        
        Args:
            session_id: Optional session ID. If None, returns all records from all sessions.
            
        Returns:
            A list of execution records.
        """
        if session_id:
            return self.session_records.get(session_id, []).copy()
        
        # Return all records from all sessions
        all_records = []
        for records in self.session_records.values():
            all_records.extend(records)
        return all_records
    
    def get_record(self, index: int, session_id: Optional[str] = None) -> Optional[Record]:
        """Get a specific execution record by index.
        
        Args:
            index: The index of the record to retrieve.
            session_id: Optional session ID. If None, uses current_session_id or searches in all records.
            
        Returns:
            The record at the specified index, or None if index is out of range.
        """
        # If session_id not provided, use current_session_id
        if session_id is None:
            session_id = self.current_session_id
        
        records = self.get_records(session_id=session_id)
        if 0 <= index < len(records):
            return records[index]
        return None
    
    def get_last_record(self, session_id: Optional[str] = None) -> Optional[Record]:
        """Get the last record for a session.
        
        Args:
            session_id: Optional session ID. If None, uses current_session_id.
            
        Returns:
            The last record for the session, or None if no records exist.
        """
        # If session_id not provided, use current_session_id
        if session_id is None:
            session_id = self.current_session_id
        
        records = self.get_records(session_id=session_id)
        if len(records) > 0:
            return records[-1]
        return None
    
    def get_record_by_id(self, record_id: int, session_id: Optional[str] = None) -> Optional[Record]:
        """Get a specific execution record by ID.
        
        Args:
            record_id: The ID of the record to retrieve.
            session_id: Optional session ID. If None, searches in all records.
            
        Returns:
            The record with the specified ID, or None if not found.
        """
        records = self.get_records(session_id=session_id)
        for record in records:
            if record.id == record_id:
                return record
        return None
    
    def get_records_by_task_id(self, task_id: str, session_id: Optional[str] = None) -> List[Record]:
        """Get all records for a specific task ID.
        
        Args:
            task_id: The task ID to filter by.
            session_id: Optional session ID. If None, searches in all sessions.
            
        Returns:
            A list of records matching the task ID.
        """
        records = self.get_records(session_id=session_id)
        return [record for record in records if record.task_id == task_id]
    
    def clear(self, session_id: Optional[str] = None) -> None:
        """Clear execution records.
        
        Args:
            session_id: Optional session ID. If None, clears all records from all sessions.
        """
        if session_id:
            if session_id in self.session_records:
                del self.session_records[session_id]
                # If clearing current session, reset current_session_id
                if session_id == self.current_session_id:
                    self.current_session_id = None
        else:
            self.session_records.clear()
            self.current_session_id = None
            self._next_id = 1
    
    def save_to_json(self, file_path: str) -> None:
        """Save all records to a JSON file.
        
        Structure:
        {
            "metadata": {
                "current_session_id": str,
                "next_id": int,
                "session_ids": [str, ...]
            },
            "sessions": {
                "session_id": [
                    {
                        "id": int,
                        "task_id": str,
                        "observation": Any,
                        "action": Any,
                        "timestamp": str
                    },
                    ...
                ],
                ...
            }
        }
        
        Args:
            file_path: Path to the JSON file where records will be saved.
            
        Raises:
            IOError: If the file cannot be written.
        """
        file_path = Path(file_path)
        
        # Create directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare metadata
        metadata = {
            "current_session_id": self.current_session_id,
            "next_id": self._next_id,
            "session_ids": list(self.session_records.keys())
        }
        
        # Prepare sessions data
        sessions = {}
        for session_id, records in self.session_records.items():
            sessions[session_id] = []
            for record in records:
                json_record = {
                    "id": record.id,
                    "task_id": record.task_id,
                    "observation": self._serialize_for_json(record.observation),
                    "action": self._serialize_for_json(record.action),
                    "timestamp": record.timestamp
                }
                sessions[session_id].append(json_record)
        
        # Prepare save data
        save_data = {
            "metadata": metadata,
            "sessions": sessions
        }
        
        # Write to JSON file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=4, ensure_ascii=False)
    
    def load_from_json(self, file_path: str) -> None:
        """Load records from a JSON file.
        
        Expected format:
        {
            "metadata": {
                "current_session_id": str,
                "next_id": int,
                "session_ids": [str, ...]
            },
            "sessions": {
                "session_id": [
                    {
                        "id": int,
                        "task_id": str,
                        "observation": Any,
                        "action": Any,
                        "timestamp": str
                    },
                    ...
                ],
                ...
            }
        }
        
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
            load_data = json.load(f)
        
        # Validate format
        if not isinstance(load_data, dict) or "metadata" not in load_data or "sessions" not in load_data:
            raise ValueError(
                f"Invalid tracer format. Expected {{'metadata': {{...}}, 'sessions': {{...}}}}, "
                f"got: {type(load_data).__name__}"
            )
        
        # Restore metadata
        metadata = load_data.get("metadata", {})
        self.current_session_id = metadata.get("current_session_id")
        self._next_id = metadata.get("next_id", 1)
        
        # Restore sessions
        self.session_records = {}
        sessions_data = load_data.get("sessions", {})
        
        for session_id, records_data in sessions_data.items():
            self.session_records[session_id] = []
            for json_record in records_data:
                record = Record(
                    id=json_record.get("id"),
                    session_id=session_id,
                    task_id=json_record.get("task_id"),
                    observation=json_record.get("observation"),
                    action=json_record.get("action"),
                    timestamp=json_record.get("timestamp")
                )
                self.session_records[session_id].append(record)
    
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
        """Return the total number of records across all sessions."""
        return sum(len(records) for records in self.session_records.values())
    
    def get_count(self, session_id: Optional[str] = None) -> int:
        """Get the number of records.
        
        Args:
            session_id: Optional session ID. If None, returns total count from all sessions.
            
        Returns:
            Number of records.
        """
        if session_id:
            return len(self.session_records.get(session_id, []))
        return sum(len(records) for records in self.session_records.values())
    
    def get_session_ids(self) -> List[str]:
        """Get all session IDs that have records.
        
        Returns:
            A list of session IDs.
        """
        return list(self.session_records.keys())
    
    def __repr__(self) -> str:
        """Return a string representation of the Tracer."""
        total_records = sum(len(records) for records in self.session_records.values())
        return f"Tracer(records={total_records}, sessions={len(self.session_records)})"
    
    def __str__(self) -> str:
        """Return a string representation of the Tracer."""
        return self.__repr__()

