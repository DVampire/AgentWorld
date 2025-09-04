"""Configuration for File System Environment."""

# Environment rules for the file system environment
environment_rules = """<environment_file_system>

<state>
The file system environment provides a persistent file system for the agent to manage files, documents, and data.
Current state includes available files, current directory, and file system information.
File system supports multiple file types: markdown, text, JSON, CSV, and PDF files.
</state>

<vision>
No vision available.
</vision>

<interaction>
The agent can perform file operations including read, write, append, replace, list, describe, validate, and export.
File size limit: 1MB per file, supported extensions: md, txt, json, csv, pdf.
All operations are executed asynchronously and return success/failure status with metadata.
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
