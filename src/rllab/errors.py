"""Domain-specific exceptions with actionable error messages."""


class RLLabError(Exception):
    """Base exception for expected RLLab failures."""


class ConfigurationError(RLLabError):
    """Raised when a checked configuration is invalid."""


class ValidationError(RLLabError):
    """Raised when an MDP, tokenizer registry, or artifact is invalid."""


class ArtifactError(RLLabError):
    """Raised when a versioned artifact cannot be safely read or published."""
