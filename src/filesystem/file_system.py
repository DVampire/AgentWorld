import os
from re import S
import shutil
from pathlib import Path
from typing import Any, List, Optional, Dict, Union

from src.filesystem.service import FileSystemService
from src.filesystem.types import FileReadRequest
from src.filesystem.exceptions import FileSystemError, InvalidPathError, PathTraversalError
from src.utils import get_file_info

class FileSystem:
	"""Enhanced file system with in-memory storage and multiple file type support"""

	def __init__(self, base_dir: Union[str, Path], create_default_files: bool = True):
		"""Initialize the file system with a base directory.
		
		Args:
			base_dir: Base directory for file operations
			create_default_files: Whether to create default files (unused, kept for compatibility)
		"""
		self.base_dir = Path(base_dir) if isinstance(base_dir, str) else base_dir
		self.base_dir.mkdir(parents=True, exist_ok=True)
  
		# New async service for actual IO; this class stays as adapter for compatibility
		self._service = FileSystemService(self.base_dir)

		self.extracted_content_count = 0

	def _validate_absolute_path(self, file_path: str) -> tuple[Path, Path]:
		"""Validate and convert absolute path to relative path within base directory.
		
		Args:
			file_path: Absolute file path to validate
			
		Returns:
			Tuple of (absolute_path, relative_path)
			
		Raises:
			InvalidPathError: If path is invalid or outside base directory
		"""
		if not file_path:
			raise InvalidPathError("File path is required")
		if not os.path.isabs(file_path):
			raise InvalidPathError("File path must be absolute")
		
		abs_path = Path(file_path).resolve()
		try:
			rel_path = abs_path.relative_to(self.base_dir.resolve())
		except ValueError:
			raise PathTraversalError(f"Path is outside base directory: {file_path}")
		
		return abs_path, rel_path

	def _format_line_range_result(self, file_path: str, content: str, start_line: int, end_line: int, total_lines: int) -> str:
		"""Format result for line range operations.
		
		Args:
			file_path: Path to the file
			content: File content
			start_line: Start line number
			end_line: End line number
			total_lines: Total number of lines in file
			
		Returns:
			Formatted string with line numbers
		"""
		lines = content.splitlines()
		start = start_line - 1 if start_line else 0
		end = end_line if end_line else total_lines
		
		if start < 0 or end > total_lines or start >= end:
			return f"Error: Invalid line range. File has {total_lines} lines."
		
		slice_lines = lines[start:end]
		result_lines = []
		for i, line in enumerate(slice_lines, start=start + 1):
			result_lines.append(f"{i:4d}: {line}")
		
		return f"File: {file_path} (disk)\nLines {start + 1}-{end} of {total_lines}:\n" + "\n".join(result_lines)


	async def read_file(self, file_path: str, start_line: Optional[int] = None, end_line: Optional[int] = None) -> str:
		"""Read file content with optional line range support, prioritizing memory over disk"""
		try:
			abs_path, rel_path = self._validate_absolute_path(file_path)
			
			# Delegate to async service
			req = FileReadRequest(path=rel_path, start_line=start_line, end_line=end_line)
			result = await self._service.read(req)
			
			if start_line is not None or end_line is not None:
				content = result.content_text or ''
				total_lines = result.total_lines if result.total_lines is not None else len(content.splitlines())
				return self._format_line_range_result(file_path, content, start_line, end_line, total_lines)
			
			return f"Read from file {file_path} (disk).\n<content>\n{result.content_text or ''}\n</content>"
			
		except (InvalidPathError, PathTraversalError, FileSystemError) as e:
			return f"Error: {str(e)}"
		except FileNotFoundError:
			return f"Error: File '{file_path}' not found."
		except PermissionError:
			return f"Error: Permission denied to read file '{file_path}'."
		except Exception as e:
			return f"Error: Could not read file '{file_path}': {str(e)}"

	async def write_file(self, file_path: str, content: str, mode: str = "w") -> str:
		"""Write content to file with mode support (w=overwrite, a=append) and sync to memory"""
		try:
			if content is None:
				raise InvalidPathError("Content is required for WRITE operation")
			if mode not in ["w", "a"]:
				raise InvalidPathError("Mode must be 'w' (overwrite) or 'a' (append)")
			
			abs_path, rel_path = self._validate_absolute_path(file_path)
			await self._service.write_text(rel_path, content, mode=mode)
			
			action = "overwritten" if mode == "w" else "appended to"
			return f"Successfully {action} file: {file_path}"
			
		except (InvalidPathError, PathTraversalError, FileSystemError) as e:
			return f"Error: {str(e)}"
		except Exception as e:
			return f"Error writing file: {str(e)}"


	async def replace_file_str(self, file_path: str, old_str: str, new_str: str, start_line: Optional[int] = None, end_line: Optional[int] = None) -> str:
		"""Replace old_str with new_str in file, optionally within a line range, and sync to memory"""
		try:
			if not old_str:
				raise InvalidPathError("Cannot replace empty string. Please provide a non-empty string to replace.")
			
			abs_path, rel_path = self._validate_absolute_path(file_path)
			replaced = await self._service.replace(rel_path, old_str, new_str, start_line=start_line, end_line=end_line)
			
			# Format range info
			if start_line is not None and end_line is not None:
				range_info = f" in lines {start_line}-{end_line}"
			elif start_line is not None:
				range_info = f" from line {start_line}"
			elif end_line is not None:
				range_info = f" up to line {end_line}"
			else:
				range_info = ""
			
			return f"Successfully replaced {replaced} occurrences of \"{old_str}\" with \"{new_str}\" in file {file_path} (disk){range_info}"
			
		except (InvalidPathError, PathTraversalError, FileSystemError) as e:
			return f"Error: {str(e)}"
		except Exception as e:
			return f"Error: Could not replace string in file '{file_path}': {str(e)}"

	async def delete_file(self, file_path: str) -> str:
		"""Delete a file from both memory and disk"""
		try:
			abs_path, rel_path = self._validate_absolute_path(file_path)
			
			# Remove on disk via service (if exists)
			if os.path.exists(file_path):
				await self._service.remove(rel_path)
				return f"Successfully deleted file: {file_path}"
			else:
				return f"Successfully deleted file from memory: {file_path}"
				
		except (InvalidPathError, PathTraversalError, FileSystemError) as e:
			return f"Error: {str(e)}"
		except Exception as e:
			return f"Error deleting file: {str(e)}"

	async def copy_file(self, src_path: str, dst_path: str) -> str:
		"""Copy a file from source to destination"""
		try:
			a_src, r_src = self._validate_absolute_path(src_path)
			a_dst, r_dst = self._validate_absolute_path(dst_path)
			
			if not os.path.exists(a_src):
				return f"Error: Source file not found: {src_path}"
			if os.path.exists(a_dst):
				return f"Error: Destination file already exists: {dst_path}"
			
			# Perform direct filesystem copy to preserve binary fidelity
			os.makedirs(os.path.dirname(a_dst), exist_ok=True)
			shutil.copy2(a_src, a_dst)
			return f"Successfully copied file from {src_path} to {dst_path}"
			
		except (InvalidPathError, PathTraversalError, FileSystemError) as e:
			return f"Error: {str(e)}"
		except Exception as e:
			return f"Error copying file: {str(e)}"

	async def move_file(self, src_path: str, dst_path: str) -> str:
		"""Move a file from source to destination"""
		try:
			a_src, r_src = self._validate_absolute_path(src_path)
			a_dst, r_dst = self._validate_absolute_path(dst_path)
			
			if not os.path.exists(a_src):
				return f"Error: Source file not found: {src_path}"
			if os.path.exists(a_dst):
				return f"Error: Destination file already exists: {dst_path}"
			
			await self._service.rename(r_src, r_dst)
			return f"Successfully moved file from {src_path} to {dst_path}"
			
		except (InvalidPathError, PathTraversalError, FileSystemError) as e:
			return f"Error: {str(e)}"
		except Exception as e:
			return f"Error moving file: {str(e)}"

	async def rename_file(self, old_path: str, new_path: str) -> str:
		"""Rename a file or directory"""
		try:
			a_old, r_old = self._validate_absolute_path(old_path)
			a_new, r_new = self._validate_absolute_path(new_path)
			
			if not os.path.exists(a_old):
				return f"Error: Source not found: {old_path}"
			if os.path.exists(a_new):
				return f"Error: Destination already exists: {new_path}"
			
			await self._service.rename(r_old, r_new)
			return f"Successfully renamed from {old_path} to {new_path}"
			
		except (InvalidPathError, PathTraversalError, FileSystemError) as e:
			return f"Error: {str(e)}"
		except Exception as e:
			return f"Error renaming file: {str(e)}"

	async def create_directory(self, dir_path: str) -> str:
		"""Create a directory"""
		try:
			abs_dir, rel = self._validate_absolute_path(dir_path)
			
			if os.path.exists(abs_dir):
				return f"Error: Directory already exists: {dir_path}"
			
			await self._service.mkdir(rel)
			return f"Successfully created directory: {dir_path}"
			
		except (InvalidPathError, PathTraversalError, FileSystemError) as e:
			return f"Error: {str(e)}"
		except Exception as e:
			return f"Error creating directory: {str(e)}"

	async def delete_directory(self, dir_path: str) -> str:
		"""Delete a directory"""
		try:
			abs_dir, rel = self._validate_absolute_path(dir_path)
			
			if not os.path.exists(abs_dir):
				return f"Error: Directory not found: {dir_path}"
			if not os.path.isdir(abs_dir):
				return f"Error: Path is not a directory: {dir_path}"
			
			await self._service.rmtree(rel)
			return f"Successfully deleted directory: {dir_path}"
			
		except (InvalidPathError, PathTraversalError, FileSystemError) as e:
			return f"Error: {str(e)}"
		except Exception as e:
			return f"Error deleting directory: {str(e)}"

	async def change_permissions(self, file_path: str, permissions: str) -> str:
		"""Change file or directory permissions"""
		try:
			if not permissions:
				raise InvalidPathError("Permissions are required for CHANGE_PERMISSIONS operation")
			
			abs_path, _ = self._validate_absolute_path(file_path)
			
			if not os.path.exists(abs_path):
				return f"Error: File not found: {file_path}"
			
			try:
				perm_octal = int(permissions, 8)
			except ValueError:
				return f"Error: Invalid permissions format '{permissions}'. Use octal format (e.g., '755', '644')"
			
			# Direct OS call (service does not wrap chmod). Retain behavior.
			os.chmod(abs_path, perm_octal)
			return f"Successfully changed permissions of {file_path} to {permissions}"
			
		except (InvalidPathError, PathTraversalError, FileSystemError) as e:
			return f"Error: {str(e)}"
		except Exception as e:
			return f"Error changing permissions: {str(e)}"
  

	async def tree_structure(self, file_path: str, max_depth: int = 3, show_hidden: bool = False, exclude_patterns: Optional[List[str]] = None, file_types: Optional[List[str]] = None) -> str:
		"""Show directory tree structure with optional filtering"""
		try:
			abs_dir, rel = self._validate_absolute_path(file_path)
			
			if not os.path.exists(file_path):
				return f"Error: Path does not exist: {file_path}"
			if not os.path.isdir(file_path):
				return f"Error: Path is not a directory: {file_path}"
			
			lines = await self._service.tree(rel, max_depth=max_depth, show_hidden=show_hidden, exclude_patterns=exclude_patterns or [], file_types=file_types)
			return f"Directory Tree Structure:\n{file_path}\n" + "\n".join(lines)
			
		except (InvalidPathError, PathTraversalError, FileSystemError) as e:
			return f"Error: {str(e)}"
		except Exception as e:
			return f"Error building tree structure: {str(e)}"
	
	
	async def search_files(self, file_path: str, query: str, search_type: str = "name", file_types: Optional[List[str]] = None, case_sensitive: bool = False, max_results: int = 50) -> str:
		"""Search for files by name or content, or search content within a single file"""
		try:
			if not query:
				raise InvalidPathError("Query is required for SEARCH operation")
			if search_type not in ["name", "content"]:
				raise InvalidPathError("Search type must be 'name' or 'content'")
			
			abs_input, rel = self._validate_absolute_path(file_path)
			
			if not os.path.exists(file_path):
				return f"Error: Path does not exist: {file_path}"
			
			by = "name" if search_type == "name" else "content"
			results = await self._service.search(rel, query=query, by=by, file_types=file_types, case_sensitive=case_sensitive, max_results=max_results)
			
			if not results:
				return f"No files found matching '{query}' in {file_path}"
			
			# If a single file content search, print matches with line numbers
			if abs_input.is_file() and by == "content" and len(results) == 1:
				matches = results[0].matches
				if not matches:
					return f"No matches found for '{query}' in file: {file_path}"
				result = f"Found {len(matches)} matches for '{query}' in file: {file_path}:\n"
				for m in matches:
					result += f"{m.line:4d}: {m.text}\n"
				return result.rstrip('\n')
			
			# Otherwise, list matching files
			result_text = f"Found {len(results)} files matching '{query}' in {file_path}:\n\n"
			for idx, item in enumerate(results, 1):
				abs_item = str(self.base_dir.joinpath(item.path).resolve())
				result_text += f"{idx}. {abs_item}\n"
			return result_text
			
		except (InvalidPathError, PathTraversalError, FileSystemError) as e:
			return f"Error: {str(e)}"
		except Exception as e:
			return f"Error searching files: {str(e)}"
	
	# Removed legacy search helpers; service-based search is used for all cases
	async def get_file_info(self, file_path: str, include_stats: bool = True) -> str:
		"""Get detailed information about a file or directory (from project.py)."""
		try:
			abs_path, rel = self._validate_absolute_path(file_path)
			
			if not os.path.exists(file_path):
				return f"Error: Path does not exist: {file_path}"
			
			info = []
			
			if include_stats:
				try:
					file_info = get_file_info(abs_path)
					info.append(f"Path: {file_info['path']}")
					info.append(f"Type: {'Directory' if file_info['is_directory'] else 'File'}")
					info.append(f"Size: {file_info['size']}")
					info.append(f"Created: {file_info['created']}")
					info.append(f"Modified: {file_info['modified']}")
					info.append(f"Permissions: {file_info['permissions']}")
     
				except Exception as e:
					info.append(f"Stats error: {str(e)}")
			
			if abs_path.is_dir():
				try:
					items = await self._service.listdir(rel)
					files = [item for item in items if (abs_path / item).is_file()]
					dirs = [item for item in items if (abs_path / item).is_dir()]
					info.append(f"Contents: {len(files)} files, {len(dirs)} directories")
				except Exception as e:
					info.append(f"Contents error: {str(e)}")
			
			return "\n".join(info)
			
		except (InvalidPathError, PathTraversalError, FileSystemError) as e:
			return f"Error: {str(e)}"
		except Exception as e:
			return f"Error getting file info: {str(e)}"
	
	async def _collect_files_info(self, path: str, file_types: Optional[List[str]], files_info: List[Dict]):
		"""Collect information about all files in the directory (from project.py)."""
		try:
			for entry in os.listdir(path):
				entry_path = os.path.join(path, entry)
				
				if os.path.isdir(entry_path):
					await self._collect_files_info(entry_path, file_types, files_info)
				else:
					# Check file type filter
					if file_types:
						file_ext = os.path.splitext(entry)[1].lower()
						if file_ext not in file_types:
							continue
					
					try:
						stat = os.stat(entry_path)
						files_info.append({
							'path': entry_path,
							'size': stat.st_size,
							'mtime': stat.st_mtime
						})
					except:
						continue
		except Exception:
			pass

	async def describe(self) -> str:
		"""Describe the file system with directory structure and file information"""
		description = f"File System Overview:\n"
		description += f"Workdir: {self.base_dir}\n\n"
		
		# Get directory tree structure
		try:
			tree_lines = await self._service.tree(Path("."), max_depth=3, show_hidden=False)
			if tree_lines:
				description += "Directory Structure:\n"
				description += f"{self.base_dir}\n"
				description += "\n".join(tree_lines)
				description += "\n\n"
		except Exception as e:
			description += f"Error building tree: {str(e)}\n\n"
		
		# Get file information summary
		try:
			all_files = await self._service.collect_all_files(Path("."), max_files=20)
			
			if all_files:
				description += "Files Summary:\n"
				for file_path in sorted(all_files):
					try:
						file_info = get_file_info(file_path)
						size_str = file_info['size']
						description += f"  📄 {file_path} ({size_str})\n"
					except Exception:
						description += f"  📄 {file_path}\n"
			else:
				description += "No files found.\n"
		except Exception as e:
			description += f"Error collecting file info: {str(e)}\n"
		
		return description.strip('\n')
