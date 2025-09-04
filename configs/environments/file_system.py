"""Configuration for File System Environment."""

# Environment rules for the file system environment
environment_rules = """<environment_file_system>

<state>
The file system environment provides a persistent file system for the agent to manage files, documents, and data.
Current state includes current directory, available files, and file information.
File system supports multiple file types: markdown, text, JSON, CSV, PDF, XLSX, DOCX, and binary files.
</state>

<vision>
No vision available.
</vision>

<interaction>
The agent can perform operations through 4 main tool categories:

1. FILE_OPERATIONS: Individual file management
   - read: Read file contents (with optional line ranges)
   - write: Write content to files (overwrite or append mode)
   - replace: Replace strings in files (with optional line ranges)
   - delete: Delete files
   - copy: Copy files from source to destination
   - move: Move files from source to destination
   - rename: Rename files or directories
   - get_info: Get detailed file information and statistics

2. DIRECTORY_OPERATIONS: Directory and system operations
   - create_dir: Create directories
   - delete_dir: Delete directories
   - tree: Show directory tree structure with customizable depth and filters
   - describe: Get comprehensive file system description and structure

3. SEARCH_OPERATIONS: File and content search
   - search: Search files by name or content with filtering options
   - Supports case-sensitive/insensitive search
   - Filter by file types and patterns
   - Configurable result limits

4. PERMISSION_OPERATIONS: File and directory permissions
   - change_permissions: Modify file/directory permissions in octal format

File size limit: 1MB per file, supported extensions: md, txt, json, csv, pdf, xlsx, docx, py, and binary files.
All operations are executed asynchronously through the environment and return success/failure status with detailed metadata.
</interaction>

</environment_file_system>
"""

# File System Environment Configuration
environment = dict(
    type="FileSystemEnvironment",
    base_dir="workdir/file_system",
    create_default_files=True,
    max_file_size=1024 * 1024,  # 1MB
)

controller = dict(
    type="FileSystemController",
    environment=environment,
    environment_rules=environment_rules,
)
