"""Provider-independent errors with separate diagnostic and UI-facing text.

``str(error)`` is for logs and debugging and may include provider response data.
Use ``code`` for application logic/localization and ``user_message`` for a safe
default UI message. Mapped SDK exceptions remain available as ``__cause__``.
"""


class LlmkitError(Exception):
    """Base exception for classified llmkit failures."""

    code = "model_error"
    user_message = "The model request failed."
    retryable = False

    def __init__(
        self, message: str = "Model request failed", *, status_code: int | None = None
    ) -> None:
        super().__init__(message)
        self.status_code = status_code


class ProviderError(LlmkitError):
    """The provider rejected a request for a reason not classified more narrowly."""

    code = "model_provider_error"
    user_message = "The model provider rejected the request."

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message, status_code=status_code)


class ResponseInterruptedError(LlmkitError):
    """A response began but its connection ended before completion."""

    code = "model_response_interrupted"
    user_message = "The model response was interrupted. Please try again."


class RetryableError(LlmkitError):
    """A temporary provider, connection, or timeout failure."""

    code = "model_temporarily_unavailable"
    user_message = "The model is temporarily unavailable. Please try again."
    retryable = True

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message, status_code=status_code)


class RateLimitError(RetryableError):
    """The provider returned HTTP 429 Too Many Requests."""

    code = "model_rate_limited"
    user_message = "The model is busy. Please wait and try again."

    def __init__(
        self, message: str = "Rate limit exceeded", retry_after: float | None = None
    ) -> None:
        super().__init__(message, status_code=429)
        self.retry_after = retry_after


class OutOfCreditsError(LlmkitError):
    """The account has insufficient credit or quota for this request."""

    code = "model_out_of_credits"
    user_message = "The model account has no available credits."

    def __init__(
        self, message: str = "Out of credits", *, status_code: int | None = None
    ) -> None:
        super().__init__(message, status_code=status_code)


class ContextLengthExceededError(LlmkitError):
    """The input exceeds the model's maximum context length."""

    code = "model_context_too_long"
    user_message = "The request is too long for this model."

    def __init__(
        self,
        message: str = "Context length exceeded",
        *,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message, status_code=status_code)


class ModelBehaviorError(LlmkitError):
    """The answer does not fit the shape requested by the application."""

    code = "model_invalid_response"
    user_message = "The model returned an invalid response."

    def __init__(self, message: str = "Unexpected model behaviour") -> None:
        super().__init__(message)


class AuthenticationError(LlmkitError):
    """The provider rejected the model credentials."""

    code = "model_authentication_failed"
    user_message = "The model credentials were rejected."

    def __init__(
        self, message: str = "Authentication failed", *, status_code: int | None = None
    ) -> None:
        super().__init__(message, status_code=status_code)


class CredentialsUnavailableError(AuthenticationError):
    """Credentials are missing or must be renewed, such as by signing in again."""

    code = "model_credentials_unavailable"
    user_message = "Model credentials are unavailable. Please sign in again."

    def __init__(self, message: str = "Credentials unavailable") -> None:
        super().__init__(message)
