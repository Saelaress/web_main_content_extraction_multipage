class HtmlDiffError(Exception):
    """Base exception for htmldiff errors."""


class InsufficientPagesWarning(UserWarning):
    """Raised when only a single page is provided for a domain."""


class PageTooLargeWarning(UserWarning):
    """Raised when a page exceeds max_elements and is truncated."""


class DiffTimeoutWarning(UserWarning):
    """Raised when xmldiff times out on a single diff."""


class EncodingWarning(UserWarning):
    """Raised when charset detection confidence is below threshold."""
