# GitHub Service for AgentWorld

This package provides a comprehensive GitHub service implementation for AgentWorld, featuring PAT (Personal Access Token) authentication, caching, and full GitHub API support.

## Features

- 🔐 **PAT Authentication**: Secure authentication using GitHub Personal Access Tokens
- 🚀 **Async Operations**: Fully asynchronous implementation using `async/await`
- 💾 **Intelligent Caching**: Multi-level caching with different TTL for different data types
- 🔄 **Rate Limiting**: Built-in rate limit handling and retry mechanisms
- 📊 **Comprehensive API**: Support for repositories, issues, PRs, files, branches, and more
- 🛡️ **Error Handling**: Robust error handling with specific exception types
- 🔍 **Search Capabilities**: Repository and issue search functionality
- 📁 **File Operations**: Read, create, update, and delete files in repositories

## Quick Start

### 1. Install Dependencies

Make sure you have the required dependencies installed:

```bash
pip install aiohttp pydantic
```

### 2. Create GitHub PAT

1. Go to GitHub Settings → Developer settings → Personal access tokens
2. Generate a new token with appropriate scopes:
   - `repo` - Full control of private repositories
   - `public_repo` - Access to public repositories
   - `user` - Read user profile data
   - `read:org` - Read organization membership

### 3. Basic Usage

```python
import asyncio
from src.environments.github_environment import GitHubEnvironment

async def main():
    # Create GitHub environment
    github_env = GitHubEnvironment(
        token="ghp_your_personal_access_token_here",
        username="your_username",  # Optional
        enable_cache=True,
        cache_ttl=300,  # 5 minutes
    )
    
    try:
        # Initialize environment
        await github_env.initialize()
        print(f"Authenticated as: {github_env.authenticated_user.login}")
        
        # Get repository information
        repo = await github_env.get_repository("microsoft", "vscode")
        print(f"Repository: {repo.full_name}")
        print(f"Stars: {repo.stargazers_count}")
        
        # List issues
        issues = await github_env.list_issues("microsoft", "vscode", per_page=5)
        for issue in issues:
            print(f"Issue #{issue.number}: {issue.title}")
        
    finally:
        await github_env.cleanup()

# Run the example
asyncio.run(main())
```

## Configuration

### Using Configuration Files

```python
from configs.environments.github import create_github_environment_from_config

# Create environment with predefined configuration
github_env = create_github_environment_from_config(
    token="your_token_here",
    config_name="production"  # development, production, testing, enterprise
)
```

### Custom Configuration

```python
from src.environments.github_environment import GitHubEnvironment

github_env = GitHubEnvironment(
    token="your_token_here",
    username="your_username",
    base_url="https://api.github.com",  # or GitHub Enterprise URL
    timeout=30,
    enable_cache=True,
    cache_ttl=300,
)
```

## API Reference

### Repository Operations

```python
# Get repository information
repo = await github_env.get_repository("owner", "repo")

# List repositories
repos = await github_env.list_repositories(owner="username")

# Create repository
new_repo = await github_env.create_repository(
    name="my-new-repo",
    description="Repository description",
    private=True
)
```

### File Operations

```python
# Get file content
file_content = await github_env.get_file_content("owner", "repo", "path/to/file.py")

# Get directory contents
contents = await github_env.get_directory_contents("owner", "repo", "src/")

# Create or update file
result = await github_env.create_or_update_file(
    owner="owner",
    repo="repo",
    path="new-file.txt",
    content="File content",
    message="Add new file"
)

# Delete file
result = await github_env.delete_file(
    owner="owner",
    repo="repo",
    path="file-to-delete.txt",
    message="Remove file",
    sha="file_sha_hash"
)
```

### Issue Operations

```python
# List issues
issues = await github_env.list_issues(
    "owner", "repo",
    state="open",
    labels=["bug", "enhancement"]
)

# Get specific issue
issue = await github_env.get_issue("owner", "repo", issue_number=123)

# Create issue
new_issue = await github_env.create_issue(
    "owner", "repo",
    title="New issue title",
    body="Issue description",
    labels=["bug"]
)
```

### Pull Request Operations

```python
# List pull requests
prs = await github_env.list_pull_requests("owner", "repo", state="open")

# Get specific pull request
pr = await github_env.get_pull_request("owner", "repo", pr_number=456)
```

### Search Operations

```python
# Search repositories
search_result = await github_env.search_repositories(
    "python machine learning",
    sort="stars",
    order="desc"
)

# Search issues
issues_result = await github_env.search_issues(
    "bug in:title language:python"
)
```

### Branch Operations

```python
# List branches
branches = await github_env.list_branches("owner", "repo")

# Get specific branch
branch = await github_env.get_branch("owner", "repo", "main")
```

## Error Handling

The service provides specific exception types for different error scenarios:

```python
from src.environments.github.exceptions import (
    AuthenticationError,
    RateLimitError,
    NotFoundError,
    ConflictError,
    ValidationError,
    APIError
)

try:
    repo = await github_env.get_repository("owner", "nonexistent-repo")
except NotFoundError:
    print("Repository not found")
except AuthenticationError:
    print("Authentication failed - check your token")
except RateLimitError as e:
    print(f"Rate limit exceeded. Reset at: {e.rate_limit_reset}")
except APIError as e:
    print(f"API error: {e}")
```

## Caching

The service includes intelligent caching with different TTL for different data types:

- **User data**: 1 hour
- **Repository data**: 30 minutes
- **File content**: 5 minutes
- **Search results**: 10 minutes

```python
# Clear specific cache type
await github_env.clear_cache("repo")

# Clear all cache
await github_env.clear_cache()

# Cleanup expired entries
expired_count = await github_env.cleanup_expired_cache()
print(f"Cleaned up {expired_count} expired entries")
```

## Rate Limiting

The service automatically handles GitHub API rate limits:

```python
# Get rate limit information
rate_limit = await github_env.get_rate_limit_info()
print(f"Remaining: {rate_limit['rate']['remaining']}")
print(f"Reset at: {rate_limit['rate']['reset']}")
```

## Health Check

```python
# Perform health check
health = await github_env.health_check()
print(f"Status: {health['status']}")
print(f"Authenticated: {health['authenticated']}")
```

## Examples

See the `examples/run_github_environment.py` file for comprehensive examples including:

- Repository operations
- File operations
- Issue management
- Search functionality
- Error handling
- Cache management

## Configuration Options

### Environment Variables

```bash
export GITHUB_TOKEN="ghp_your_personal_access_token_here"
export GITHUB_USERNAME="your_username"  # Optional
export GITHUB_BASE_URL="https://api.github.com"  # Optional
```

### Configuration Profiles

The service includes predefined configuration profiles:

- **development**: Longer timeouts, moderate caching
- **production**: Optimized for production use
- **testing**: Minimal caching, fast timeouts
- **enterprise**: Configured for GitHub Enterprise

## Security Considerations

1. **Token Security**: Never commit PAT tokens to version control
2. **Scope Limitation**: Use minimal required scopes for your use case
3. **Token Rotation**: Regularly rotate your PAT tokens
4. **Environment Variables**: Store tokens in environment variables
5. **Network Security**: Use HTTPS for all API communications

## Troubleshooting

### Common Issues

1. **Authentication Error**: Verify your PAT token and scopes
2. **Rate Limit Exceeded**: Implement exponential backoff or reduce request frequency
3. **Repository Not Found**: Check repository name and permissions
4. **Network Timeout**: Increase timeout value or check network connectivity

### Debug Mode

Enable debug logging to troubleshoot issues:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

When contributing to the GitHub service:

1. Follow the existing code structure
2. Add appropriate error handling
3. Include type hints
4. Add tests for new functionality
5. Update documentation

## License

This package is part of AgentWorld and follows the same license terms.
