"""GitHub service exceptions."""


class GitHubError(Exception):
    """Base GitHub service exception."""
    pass


class AuthenticationError(GitHubError):
    """GitHub authentication error."""
    pass


class NotFoundError(GitHubError):
    """GitHub resource not found error."""
    pass


class GitError(GitHubError):
    """Git operation error."""
    pass


class RepositoryError(GitHubError):
    """Repository operation error."""
    pass
