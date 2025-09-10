"""GitHub environment configuration."""

from src.environments.github_environment import GitHubEnvironment
from src.environments.github import GitHubAuth, GitHubConfig


def create_github_environment(
    token: str,
    username: str = None,
    base_url: str = "https://api.github.com",
    timeout: int = 30,
    enable_cache: bool = True,
    cache_ttl: int = 300,
    default_per_page: int = 30,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    enable_rate_limit: bool = True,
    **kwargs
) -> GitHubEnvironment:
    """Create and configure GitHub environment.
    
    Args:
        token: GitHub Personal Access Token (PAT)
        username: GitHub username (optional, will be fetched if not provided)
        base_url: GitHub API base URL (default: https://api.github.com)
        timeout: Request timeout in seconds (default: 30)
        enable_cache: Enable caching for API responses (default: True)
        cache_ttl: Cache time-to-live in seconds (default: 300)
        default_per_page: Default number of items per page (default: 30)
        max_retries: Maximum number of retries for failed requests (default: 3)
        retry_delay: Delay between retries in seconds (default: 1.0)
        enable_rate_limit: Enable rate limiting (default: True)
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured GitHub environment instance
        
    Example:
        ```python
        # Create GitHub environment with PAT
        github_env = create_github_environment(
            token="ghp_your_personal_access_token_here",
            username="your_username",  # Optional
            enable_cache=True,
            cache_ttl=600,  # 10 minutes
        )
        
        # Initialize and use
        await github_env.initialize()
        
        # Get repository information
        repo = await github_env.get_repository("owner", "repo_name")
        
        # List issues
        issues = await github_env.list_issues("owner", "repo_name")
        
        # Cleanup
        await github_env.cleanup()
        ```
    """
    
    # Create authentication configuration
    auth = GitHubAuth(
        token=token,
        username=username,
        base_url=base_url,
        timeout=timeout
    )
    
    # Create service configuration
    config = GitHubConfig(
        auth=auth,
        default_per_page=default_per_page,
        max_retries=max_retries,
        retry_delay=retry_delay,
        cache_ttl=cache_ttl,
        enable_cache=enable_cache,
        enable_rate_limit=enable_rate_limit,
        **kwargs
    )
    
    # Create and return environment
    return GitHubEnvironment(
        token=token,
        username=username,
        base_url=base_url,
        timeout=timeout,
        enable_cache=enable_cache,
        cache_ttl=cache_ttl,
        **kwargs
    )


# Example configuration for different use cases
GITHUB_CONFIGS = {
    "development": {
        "timeout": 60,
        "enable_cache": True,
        "cache_ttl": 300,  # 5 minutes
        "max_retries": 3,
        "retry_delay": 1.0,
    },
    
    "production": {
        "timeout": 30,
        "enable_cache": True,
        "cache_ttl": 600,  # 10 minutes
        "max_retries": 5,
        "retry_delay": 2.0,
    },
    
    "testing": {
        "timeout": 10,
        "enable_cache": False,
        "cache_ttl": 0,
        "max_retries": 1,
        "retry_delay": 0.5,
    },
    
    "enterprise": {
        "base_url": "https://github.your-company.com/api/v3",
        "timeout": 45,
        "enable_cache": True,
        "cache_ttl": 900,  # 15 minutes
        "max_retries": 3,
        "retry_delay": 1.5,
    }
}


def get_github_config(config_name: str = "development") -> dict:
    """Get predefined GitHub configuration.
    
    Args:
        config_name: Configuration name (development, production, testing, enterprise)
        
    Returns:
        Configuration dictionary
        
    Raises:
        ValueError: If config_name is not found
    """
    if config_name not in GITHUB_CONFIGS:
        raise ValueError(f"Unknown config name: {config_name}. Available: {list(GITHUB_CONFIGS.keys())}")
    
    return GITHUB_CONFIGS[config_name].copy()


def create_github_environment_from_config(
    token: str,
    config_name: str = "development",
    **overrides
) -> GitHubEnvironment:
    """Create GitHub environment from predefined configuration.
    
    Args:
        token: GitHub Personal Access Token
        config_name: Configuration name (development, production, testing, enterprise)
        **overrides: Configuration overrides
        
    Returns:
        Configured GitHub environment
    """
    config = get_github_config(config_name)
    config.update(overrides)
    
    return create_github_environment(token=token, **config)
