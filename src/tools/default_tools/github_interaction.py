"""
GitHub Interaction Tool - A comprehensive tool for various GitHub operations
Supports PAT (Personal Access Token) authentication, can manage multiple repositories, and perform various Git operations
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dotenv import load_dotenv
from langchain_core.tools import BaseTool
from pydantic import Field, BaseModel

from src.tools.base import ToolResponse

# Check dependencies
try:
    from github import Github, GithubException
except ImportError:
    raise ImportError("PyGithub is required. Install with: pip install PyGithub")

try:
    import git
except ImportError:
    raise ImportError("GitPython is required. Install with: pip install GitPython")

load_dotenv(verbose=True)


class GitHubRepoArgs(BaseModel):
    """GitHub repository operation arguments"""
    operation: str = Field(description="Operation type: init, create, clone, delete, branch_create, branch_delete, branch_copy, commit, push, fetch, pull, status, log")
    repo_name: Optional[str] = Field(default=None, description="Repository name (format: owner/repo)")
    local_path: Optional[str] = Field(default=None, description="Local path")
    branch_name: Optional[str] = Field(default=None, description="Branch name")
    source_branch: Optional[str] = Field(default=None, description="Source branch name (for branch copying)")
    commit_message: Optional[str] = Field(default=None, description="Commit message")
    files: Optional[List[str]] = Field(default=None, description="List of files to commit")
    description: Optional[str] = Field(default=None, description="Repository description")
    private: bool = Field(default=False, description="Whether the repository is private")
    auto_init: bool = Field(default=True, description="Whether to auto-initialize the repository")


class GitHubInteractionTool(BaseTool):
    """GitHub interaction tool - supports various GitHub and Git operations"""
    
    name: str = "github_interaction"
    description: str = (
        "GitHub interaction tool that supports various GitHub and Git operations. "
        "Can manage multiple repositories, perform Git operations such as creating, cloning, deleting repositories, "
        "branch management, commit, push, pull and other operations."
    )
    args_schema: type[BaseModel] = GitHubRepoArgs
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.github_token = os.getenv("GITHUB_TOKEN")
        if not self.github_token:
            raise ValueError("GITHUB_TOKEN environment variable is required")
        
        try:
            self.github = Github(self.github_token)
            self.user = self.github.get_user()
            print(f"✅ GitHub authentication successful, user: {self.user.login}")
        except Exception as e:
            raise ValueError(f"GitHub authentication failed: {e}")
    
    def _run(self, **kwargs) -> ToolResponse:
        """Synchronously execute GitHub operations"""
        try:
            return self._execute_operation(**kwargs)
        except Exception as e:
            return ToolResponse(content=f"Operation failed: {str(e)}")
    
    async def _arun(self, **kwargs) -> ToolResponse:
        """Asynchronously execute GitHub operations"""
        return self._run(**kwargs)
    
    def _execute_operation(self, **kwargs) -> ToolResponse:
        """Execute specific GitHub operations"""
        operation = kwargs.get("operation", "").lower()
        
        if operation == "init":
            return self._init_repo(**kwargs)
        elif operation == "create":
            return self._create_repo(**kwargs)
        elif operation == "clone":
            return self._clone_repo(**kwargs)
        elif operation == "delete":
            return self._delete_repo(**kwargs)
        elif operation == "branch_create":
            return self._create_branch(**kwargs)
        elif operation == "branch_delete":
            return self._delete_branch(**kwargs)
        elif operation == "branch_copy":
            return self._copy_branch(**kwargs)
        elif operation == "commit":
            return self._commit_changes(**kwargs)
        elif operation == "push":
            return self._push_changes(**kwargs)
        elif operation == "fetch":
            return self._fetch_changes(**kwargs)
        elif operation == "pull":
            return self._pull_changes(**kwargs)
        elif operation == "status":
            return self._get_status(**kwargs)
        elif operation == "log":
            return self._get_log(**kwargs)
        else:
            return ToolResponse(content=f"Unsupported operation: {operation}")
    
    def _init_repo(self, **kwargs) -> ToolResponse:
        """Initialize local Git repository"""
        local_path = kwargs.get("local_path", ".")
        
        try:
            if not os.path.exists(local_path):
                os.makedirs(local_path)
            
            # Check if it's already a Git repository
            if os.path.exists(os.path.join(local_path, ".git")):
                return ToolResponse(content=f"Path {local_path} is already a Git repository")
            
            # Initialize Git repository
            repo = git.Repo.init(local_path)
            
            # Create README file
            readme_path = os.path.join(local_path, "README.md")
            if not os.path.exists(readme_path):
                with open(readme_path, "w") as f:
                    f.write(f"# {os.path.basename(local_path)}\n\nThis is a newly initialized Git repository.\n")
            
            # Add and commit initial files
            repo.index.add(["README.md"])
            repo.index.commit("Initial commit")
            
            return ToolResponse(content=f"✅ Successfully initialized Git repository: {local_path}")
            
        except Exception as e:
            return ToolResponse(content=f"Failed to initialize repository: {str(e)}")
    
    def _create_repo(self, **kwargs) -> ToolResponse:
        """Create new repository on GitHub"""
        repo_name = kwargs.get("repo_name")
        description = kwargs.get("description", "")
        private = kwargs.get("private", False)
        auto_init = kwargs.get("auto_init", True)
        
        if not repo_name:
            return ToolResponse(content="Error: Repository name is required")
        
        try:
            # Create GitHub repository
            repo = self.user.create_repo(
                name=repo_name,
                description=description,
                private=private,
                auto_init=auto_init
            )
            
            result = f"✅ Successfully created repository: {repo.full_name}\n"
            result += f"URL: {repo.html_url}\n"
            result += f"SSH: {repo.ssh_url}\n"
            result += f"HTTPS: {repo.clone_url}"
            
            return ToolResponse(content=result)
            
        except GithubException as e:
            if e.status == 422:
                return ToolResponse(content=f"Repository {repo_name} already exists")
            return ToolResponse(content=f"Failed to create repository: {str(e)}")
    
    def _clone_repo(self, **kwargs) -> ToolResponse:
        """Clone GitHub repository to local"""
        repo_name = kwargs.get("repo_name")
        local_path = kwargs.get("local_path")
        
        if not repo_name:
            return ToolResponse(content="Error: Repository name is required")
        
        if not local_path:
            local_path = os.path.join(os.getcwd(), repo_name.split("/")[-1])
        
        try:
            # Get repository information
            if "/" in repo_name:
                repo = self.github.get_repo(repo_name)
            else:
                repo = self.user.get_repo(repo_name)
            
            # Clone repository
            git.Repo.clone_from(repo.clone_url, local_path)
            
            return ToolResponse(content=f"✅ Successfully cloned repository to: {local_path}")
            
        except Exception as e:
            return ToolResponse(content=f"Failed to clone repository: {str(e)}")
    
    def _delete_repo(self, **kwargs) -> ToolResponse:
        """Delete GitHub repository"""
        repo_name = kwargs.get("repo_name")
        
        if not repo_name:
            return ToolResponse(content="Error: Repository name is required")
        
        try:
            # Get repository
            if "/" in repo_name:
                repo = self.github.get_repo(repo_name)
            else:
                repo = self.user.get_repo(repo_name)
            
            # Delete repository
            repo.delete()
            
            return ToolResponse(content=f"✅ Successfully deleted repository: {repo_name}")
            
        except Exception as e:
            return ToolResponse(content=f"Failed to delete repository: {str(e)}")
    
    def _create_branch(self, **kwargs) -> ToolResponse:
        """Create new branch"""
        repo_name = kwargs.get("repo_name")
        branch_name = kwargs.get("branch_name")
        local_path = kwargs.get("local_path")
        
        if not branch_name:
            return ToolResponse(content="Error: Branch name is required")
        
        try:
            if local_path and os.path.exists(local_path):
                # Local operation
                repo = git.Repo(local_path)
                repo.git.checkout("-b", branch_name)
                return ToolResponse(content=f"✅ Successfully created local branch: {branch_name}")
            else:
                # GitHub operation
                if not repo_name:
                    return ToolResponse(content="Error: Repository name is required")
                
                repo = self.github.get_repo(repo_name)
                main_branch = repo.default_branch
                main_ref = repo.get_branch(main_branch)
                
                repo.create_git_ref(
                    ref=f"refs/heads/{branch_name}",
                    sha=main_ref.commit.sha
                )
                
                return ToolResponse(content=f"✅ Successfully created remote branch: {branch_name}")
                
        except Exception as e:
            return ToolResponse(content=f"Failed to create branch: {str(e)}")
    
    def _delete_branch(self, **kwargs) -> ToolResponse:
        """Delete branch"""
        repo_name = kwargs.get("repo_name")
        branch_name = kwargs.get("branch_name")
        local_path = kwargs.get("local_path")
        
        if not branch_name:
            return ToolResponse(content="Error: Branch name is required")
        
        try:
            if local_path and os.path.exists(local_path):
                # Local operation
                repo = git.Repo(local_path)
                if branch_name == repo.active_branch.name:
                    repo.git.checkout("main")  # Switch to main branch
                repo.git.branch("-d", branch_name)
                return ToolResponse(content=f"✅ Successfully deleted local branch: {branch_name}")
            else:
                # GitHub operation
                if not repo_name:
                    return ToolResponse(content="Error: Repository name is required")
                
                repo = self.github.get_repo(repo_name)
                ref = repo.get_git_ref(f"heads/{branch_name}")
                ref.delete()
                
                return ToolResponse(content=f"✅ Successfully deleted remote branch: {branch_name}")
                
        except Exception as e:
            return ToolResponse(content=f"Failed to delete branch: {str(e)}")
    
    def _copy_branch(self, **kwargs) -> ToolResponse:
        """Copy branch"""
        repo_name = kwargs.get("repo_name")
        source_branch = kwargs.get("source_branch")
        branch_name = kwargs.get("branch_name")
        local_path = kwargs.get("local_path")
        
        if not source_branch or not branch_name:
            return ToolResponse(content="Error: Source branch and target branch names are required")
        
        try:
            if local_path and os.path.exists(local_path):
                # Local operation
                repo = git.Repo(local_path)
                repo.git.checkout(source_branch)
                repo.git.checkout("-b", branch_name)
                return ToolResponse(content=f"✅ Successfully created branch {branch_name} from {source_branch}")
            else:
                # GitHub operation
                if not repo_name:
                    return ToolResponse(content="Error: Repository name is required")
                
                repo = self.github.get_repo(repo_name)
                source_ref = repo.get_branch(source_branch)
                
                repo.create_git_ref(
                    ref=f"refs/heads/{branch_name}",
                    sha=source_ref.commit.sha
                )
                
                return ToolResponse(content=f"✅ Successfully created remote branch {branch_name} from {source_branch}")
                
        except Exception as e:
            return ToolResponse(content=f"Failed to copy branch: {str(e)}")
    
    def _commit_changes(self, **kwargs) -> ToolResponse:
        """Commit changes"""
        local_path = kwargs.get("local_path", ".")
        commit_message = kwargs.get("commit_message", "Update")
        files = kwargs.get("files", [])
        
        try:
            repo = git.Repo(local_path)
            
            if files:
                # Add specified files
                for file in files:
                    repo.index.add([file])
            else:
                # Add all changes
                repo.git.add(".")
            
            # Check if there are changes
            if not repo.index.diff("HEAD"):
                return ToolResponse(content="No changes to commit")
            
            # Commit changes
            repo.index.commit(commit_message)
            
            return ToolResponse(content=f"✅ Successfully committed changes: {commit_message}")
            
        except Exception as e:
            return ToolResponse(content=f"Failed to commit: {str(e)}")
    
    def _push_changes(self, **kwargs) -> ToolResponse:
        """Push changes"""
        local_path = kwargs.get("local_path", ".")
        branch_name = kwargs.get("branch_name")
        
        try:
            repo = git.Repo(local_path)
            
            if branch_name:
                repo.git.push("origin", branch_name)
            else:
                repo.git.push()
            
            return ToolResponse(content="✅ Successfully pushed changes")
            
        except Exception as e:
            return ToolResponse(content=f"Failed to push: {str(e)}")
    
    def _fetch_changes(self, **kwargs) -> ToolResponse:
        """Fetch remote changes"""
        local_path = kwargs.get("local_path", ".")
        
        try:
            repo = git.Repo(local_path)
            repo.git.fetch()
            
            return ToolResponse(content="✅ Successfully fetched remote changes")
            
        except Exception as e:
            return ToolResponse(content=f"Failed to fetch: {str(e)}")
    
    def _pull_changes(self, **kwargs) -> ToolResponse:
        """Pull remote changes"""
        local_path = kwargs.get("local_path", ".")
        branch_name = kwargs.get("branch_name")
        
        try:
            repo = git.Repo(local_path)
            
            if branch_name:
                repo.git.pull("origin", branch_name)
            else:
                repo.git.pull()
            
            return ToolResponse(content="✅ Successfully pulled remote changes")
            
        except Exception as e:
            return ToolResponse(content=f"Failed to pull: {str(e)}")
    
    def _get_status(self, **kwargs) -> ToolResponse:
        """Get repository status"""
        local_path = kwargs.get("local_path", ".")
        
        try:
            repo = git.Repo(local_path)
            status = repo.git.status()
            
            return ToolResponse(content=f"📊 Repository status:\n{status}")
            
        except Exception as e:
            return ToolResponse(content=f"Failed to get status: {str(e)}")
    
    def _get_log(self, **kwargs) -> ToolResponse:
        """Get commit history"""
        local_path = kwargs.get("local_path", ".")
        limit = kwargs.get("limit", 10)
        
        try:
            repo = git.Repo(local_path)
            commits = list(repo.iter_commits(max_count=limit))
            
            log_info = "📋 Commit history:\n"
            for commit in commits:
                log_info += f"{commit.hexsha[:8]} - {commit.message.strip()} ({commit.author.name})\n"
            
            return ToolResponse(content=log_info)
            
        except Exception as e:
            return ToolResponse(content=f"Failed to get log: {str(e)}")
    
    def get_tool_config(self) -> Dict[str, Any]:
        """Get tool configuration"""
        return {
            "name": self.name,
            "description": self.description,
            "args_schema": self.args_schema,
            "type": "github_interaction"
        }