"""
Internal helper: cloud model name → provider inference.

Design:
  - Pure data + pure functions — no project imports, no side effects.
  - Single source of truth for "context_model string → provider key" logic,
    consumed by both validator.py and generator.py.
  - Leading underscore = internal API, not part of ragfactory's public surface.
"""

from __future__ import annotations

# ─── Cloud provider prefix table ──────────────────────────────────────────────
# Each entry: (prefixes_tuple, provider_key).
# Ordered from most-specific to least-specific to prevent prefix shadowing.
# provider_key must match the keys used in PROVIDER_ENV_VAR below.

_CLOUD_PROVIDER_PREFIXES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("gpt-", "o1-", "o1", "text-embedding-", "text-"), "openai"),
    (("claude-",),                                        "anthropic"),
    (("command",),                                        "cohere_llm"),
    (("gemini-",),                                        "gemini"),
    (("mistral-", "mistral"),                             "mistral"),
    (("bedrock/",),                                       "bedrock"),
)

# ─── Env-var lookup ────────────────────────────────────────────────────────────
# Maps provider_key → the primary env var name users must set.
# Used by both validator (warning messages) and generator (.env.example output).

PROVIDER_ENV_VAR: dict[str, str] = {
    "openai":     "OPENAI_API_KEY",
    "anthropic":  "ANTHROPIC_API_KEY",
    "cohere_llm": "COHERE_API_KEY",
    "gemini":     "GOOGLE_API_KEY",
    "mistral":    "MISTRAL_API_KEY",
    "bedrock":    "AWS_BEARER_TOKEN_BEDROCK",
}

# Providers for which ragfactory can automatically scaffold contextual chunking
# environment variables in the generated project.
_AUTOMATIC_CONTEXTUAL_PROVIDER_SCAFFOLDING: frozenset[str] = frozenset({
    "openai",
    "anthropic",
    "cohere_llm",
    "gemini",
})


# ─── Public functions ──────────────────────────────────────────────────────────


def infer_context_model_provider(model_name: str) -> str | None:
    """
    Return the provider key for a known cloud model name, or None if unrecognised.

    Args:
        model_name: The model identifier string (e.g. "gpt-4o-mini",
                    "claude-3-haiku-20240307", "gemini-1.5-flash", "llama3.2").

    Returns:
        A provider key string (matches keys in PROVIDER_ENV_VAR), or None when
        the model is not a recognised cloud provider. None callers should treat
        the model as either local or an unknown/custom provider.

    Examples:
        >>> infer_context_model_provider("gpt-4o-mini")
        'openai'
        >>> infer_context_model_provider("claude-3-haiku-20240307")
        'anthropic'
        >>> infer_context_model_provider("gemini-1.5-flash")
        'gemini'
        >>> infer_context_model_provider("command-r-plus")
        'cohere_llm'
        >>> infer_context_model_provider("llama3.2")
        None
    """
    for prefixes, provider in _CLOUD_PROVIDER_PREFIXES:
        if any(model_name.startswith(p) for p in prefixes):
            return provider
    return None


def supports_contextual_provider_scaffolding(provider: str) -> bool:
    """Return True when ragfactory scaffolds env vars for this provider."""
    return provider in _AUTOMATIC_CONTEXTUAL_PROVIDER_SCAFFOLDING


def is_probably_local_model(model_name: str) -> bool:
    """
    Heuristic: return True if the model name looks like a local Ollama model.

    A model is considered 'probably local' when:
      - It has no recognised cloud provider prefix (infer_context_model_provider → None)
      - It contains no path separator '/' (HuggingFace: 'org/model')
      - It contains no port/tag separator ':' (Ollama: 'model:tag')

    Unknown cloud-like models with '/' or ':' return False — we don't know what
    they are but we shouldn't warn that they're slow local models.
    """
    return (
        infer_context_model_provider(model_name) is None
        and "/" not in model_name
        and ":" not in model_name
    )
