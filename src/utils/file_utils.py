from typing import Dict, Any
from pathlib import Path
from datetime import datetime

def format_size(size_bytes: int) -> str:
    """Format file size in human readable format (from project.py)."""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"

def get_file_info(file_path: str) -> Dict[str, Any]:
    """Get file information."""
    file_path = Path(file_path)
    
    info = {}
    file_stats = file_path.stat()

    info["path"] = str(file_path.resolve().absolute())
    info["size"] = format_size(file_stats.st_size)
    info["created"] = datetime.fromtimestamp(file_stats.st_ctime).strftime("%Y-%m-%d %H:%M:%S")
    info["modified"] = datetime.fromtimestamp(file_stats.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    info["accessed"] = datetime.fromtimestamp(file_stats.st_atime).strftime("%Y-%m-%d %H:%M:%S")
    info["permissions"] = oct(file_stats.st_mode)[-3:]
    info["is_directory"] = file_path.is_dir()
    info["is_file"] = file_path.is_file()
    info["is_symlink"] = file_path.is_symlink()
    
    return info