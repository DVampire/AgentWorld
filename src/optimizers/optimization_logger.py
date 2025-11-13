"""
Optimization log manager module.
Provides utilities for recording optimization progress.
"""

import os
from datetime import datetime

from src.logger import logger


class OptimizationLogger:
    """Optimization log manager."""
    
    def __init__(self, log_dir: str, optimizer_name: str = None):
        """
        Initialize the log manager.

        Args:
            log_dir: Directory where log files are stored.
            optimizer_name: Optional optimizer name used to label the log file.
        """
        self.log_dir = log_dir
        self.optimizer_name = optimizer_name
        self.log_file = None
        self.log_file_path = None
        self._setup_log_file()
    
    def _setup_log_file(self):
        """Configure the log file."""
        # Ensure the log directory exists.
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Create a log file name that includes a timestamp and optimizer name.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.optimizer_name:
            # Convert the optimizer class name to lowercase and remove the "Optimizer" suffix if present.
            # Example: TextGradOptimizer -> textgrad, ReflectionOptimizer -> reflection.
            name = self.optimizer_name.lower().replace("optimizer", "").strip()
            # Fallback to a default name if the result becomes empty (e.g., class name is simply "Optimizer").
            if not name:
                name = "default"
            filename = f"optimization_{name}_{timestamp}.log"
        else:
            filename = f"optimization_{timestamp}.log"
        self.log_file_path = os.path.join(self.log_dir, filename)
        
        # Open the log file for writing with UTF-8 encoding.
        self.log_file = open(self.log_file_path, "w", encoding="utf-8")
        
        logger.info(f"| 📝 Optimization log will be saved to: {self.log_file_path}")
    
    def write(self, message: str):
        """
        Write a log entry to the optimization log file.

        Args:
            message: Message to write (without a trailing newline).
        """
        if self.log_file:
            # Remove the table-style prefix so the plain-text file stays readable.
            clean_message = message.replace("| ", "").strip()
            self.log_file.write(clean_message + "\n")
            self.log_file.flush()  # Flush immediately to disk.
    
    def close(self):
        """Close the optimization log file."""
        if self.log_file:
            self.log_file.close()
            self.log_file = None

