class LLMifyError(Exception):
    """Base exception for all llmify errors."""


class RetryableError(LLMifyError):
    """Raised when a request failed but may succeed if retried (e.g. 500/503, timeouts)."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class RateLimitError(RetryableError):
    """Raised when the provider returns HTTP 429 Too Many Requests."""

    def __init__(
        self, message: str = "Rate limit exceeded", retry_after: float | None = None
    ):
        super().__init__(message, status_code=429)
        self.retry_after = retry_after


class OutOfCreditsError(LLMifyError):
    """Raised when the account has insufficient credits / quota to complete the request."""

    def __init__(self, message: str = "Out of credits"):
        super().__init__(message)


class ContextLengthExceededError(LLMifyError):
    """Raised when the input exceeds the model's maximum context length."""

    def __init__(self, message: str = "Context length exceeded"):
        super().__init__(message)


class ModelBehaviorError(LLMifyError):
    """Raised when the model's answer does not fit the shape it was asked for.

    A structured call that came back without the output tool, or with arguments
    that fail validation, is the model's mistake rather than the transport's, so
    it is not retryable.
    """

    def __init__(self, message: str = "Unexpected model behaviour"):
        super().__init__(message)


class AuthenticationError(LLMifyError):
    """Raised when the API key or credentials are invalid or missing."""

    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message)


class CredentialsUnavailableError(AuthenticationError):
    """Raised when credentials were rejected or cannot be obtained, and new ones must be acquired (e.g. a re-login)."""

    def __init__(self, message: str = "Credentials unavailable"):
        super().__init__(message)
