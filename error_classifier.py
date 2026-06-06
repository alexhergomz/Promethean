"""Classify API errors into actionable categories with recovery hints."""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ErrorCategory(Enum):
    AUTH = "auth"
    BILLING = "billing"
    RATE_LIMIT = "rate_limit"
    CONTEXT_OVERFLOW = "context_overflow"
    MODEL_NOT_FOUND = "model_not_found"
    OVERLOADED = "overloaded"
    CONNECTION = "connection"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


@dataclass
class ClassifiedError:
    category: ErrorCategory
    retryable: bool
    should_compress: bool  # compress context before retry
    backoff_multiplier: float  # multiplied with base backoff
    hint: str  # user-facing actionable message


# ── Patterns (compiled once) ─────────────────────────────────────────────

_PATTERNS: list[tuple[ErrorCategory, re.Pattern]] = [
    (ErrorCategory.AUTH, re.compile(
        r"auth|401|invalid.{0,20}(api.?key|token|credential)|unauthorized|forbidden|403",
        re.IGNORECASE)),
    (ErrorCategory.BILLING, re.compile(
        r"insufficient.{0,20}(quota|balance|credit|fund)|billing|payment|402",
        re.IGNORECASE)),
    (ErrorCategory.RATE_LIMIT, re.compile(
        r"rate.?limit|too.?many.?requests|429|throttl",
        re.IGNORECASE)),
    (ErrorCategory.CONTEXT_OVERFLOW, re.compile(
        r"context.?(length|window|size)|too.?many.?tokens|input.?is.?too.?long|"
        r"prompt.?is.?too.?long|request.?too.?large|token.?limit|max.?context|"
        # llama-server-specific
        r"exceeds.?(the.?)?available.?context|exceed_context_size_error|"
        r"prompt.?(is|exceeds).?too|n_prompt_token|"
        # Anthropic-specific 'input length' phrasing
        r"input.?length.?and.?max_tokens.?(?:exceed|larger)|"
        # MiniMax: base_resp.status_code 1039 = token limit. The platform
        # docs sometimes phrase this as 'tokens length exceeded'. Match the
        # raw code too — some SDK wrappers stringify the JSON as-is.
        r"\b1039\b|tokens?.?length.?exceed|base_?resp.*1039",
        re.IGNORECASE)),
    (ErrorCategory.MODEL_NOT_FOUND, re.compile(
        r"model.{0,20}not.?found|does.?not.?exist|unknown.?model|404.{0,30}model|"
        r"no.?such.?model",
        re.IGNORECASE)),
    (ErrorCategory.OVERLOADED, re.compile(
        r"overloaded|capacity|503|service.?unavailable|server.?busy",
        re.IGNORECASE)),
    (ErrorCategory.TIMEOUT, re.compile(
        r"timeout|timed?.?out|deadline.?exceeded|408",
        re.IGNORECASE)),
    (ErrorCategory.CONNECTION, re.compile(
        r"connect|refused|unreachable|dns|network|ECONNR|broken.?pipe|reset.?by.?peer",
        re.IGNORECASE)),
]

_HINTS = {
    ErrorCategory.AUTH:
        "Check your API key: /config or set the appropriate env var "
        "(ANTHROPIC_API_KEY, OPENAI_API_KEY, etc.)",
    ErrorCategory.BILLING:
        "Insufficient API credits. Check your billing at your provider's dashboard.",
    ErrorCategory.RATE_LIMIT:
        "Rate limited by the API. Will retry with backoff.",
    ErrorCategory.CONTEXT_OVERFLOW:
        "Context window exceeded. Compacting conversation and retrying.",
    ErrorCategory.MODEL_NOT_FOUND:
        "Model not found. Check available models with /model",
    ErrorCategory.OVERLOADED:
        "API server is overloaded. Will retry with backoff.",
    ErrorCategory.TIMEOUT:
        "Request timed out. Will retry.",
    ErrorCategory.CONNECTION:
        "Network error — check your internet connection or the API endpoint URL.",
    ErrorCategory.UNKNOWN:
        "An unexpected error occurred.",
}


def classify(exc: Exception) -> ClassifiedError:
    """Classify an exception into an actionable error category."""
    err_str = str(exc)
    err_cls = type(exc).__name__

    # Check exception class name for quick classification
    cls_lower = err_cls.lower()
    if "ratelimit" in cls_lower:
        cat = ErrorCategory.RATE_LIMIT
    elif "authentication" in cls_lower or "auth" in cls_lower:
        cat = ErrorCategory.AUTH
    elif isinstance(exc, (ConnectionError, OSError)):
        cat = ErrorCategory.CONNECTION
    elif isinstance(exc, TimeoutError):
        cat = ErrorCategory.TIMEOUT
    else:
        # Fall back to pattern matching on error message
        cat = ErrorCategory.UNKNOWN
        for category, pattern in _PATTERNS:
            if pattern.search(err_str) or pattern.search(err_cls):
                cat = category
                break

    # Check urllib errors
    try:
        import urllib.error
        if isinstance(exc, urllib.error.URLError):
            cat = ErrorCategory.CONNECTION
        elif isinstance(exc, urllib.error.HTTPError):
            code = exc.code
            if code == 401 or code == 403:
                cat = ErrorCategory.AUTH
            elif code == 402:
                cat = ErrorCategory.BILLING
            elif code == 404:
                cat = ErrorCategory.MODEL_NOT_FOUND
            elif code == 429:
                cat = ErrorCategory.RATE_LIMIT
            elif code == 503:
                cat = ErrorCategory.OVERLOADED
    except ImportError:
        pass

    # Build recovery hints per category
    retryable = cat not in (ErrorCategory.AUTH, ErrorCategory.BILLING,
                            ErrorCategory.MODEL_NOT_FOUND)
    should_compress = cat == ErrorCategory.CONTEXT_OVERFLOW
    backoff_multiplier = 3.0 if cat in (ErrorCategory.RATE_LIMIT,
                                         ErrorCategory.OVERLOADED) else 1.0

    hint = _HINTS.get(cat, _HINTS[ErrorCategory.UNKNOWN])
    if cat == ErrorCategory.CONNECTION and ("ollama" in err_str.lower()
            or "localhost" in err_str.lower() or "11434" in err_str):
        hint = "Cannot connect to Ollama. Is it running? Start with: ollama serve"

    return ClassifiedError(
        category=cat,
        retryable=retryable,
        should_compress=should_compress,
        backoff_multiplier=backoff_multiplier,
        hint=hint,
    )
