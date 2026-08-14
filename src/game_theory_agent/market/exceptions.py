"""Domain exceptions for the market environment."""


class MarketError(Exception):
    """Base exception for market-domain failures."""


class ConfigError(MarketError):
    """Raised when a market configuration is invalid."""


class ActionValidationError(MarketError):
    """Raised when an action is not schema-valid."""


class JointActionError(MarketError):
    """Raised when a joint action is incomplete or contains invalid entries."""


class StateInvariantError(MarketError):
    """Raised when a market state violates a required invariant."""


class EpisodeCompleteError(MarketError):
    """Raised when a transition is requested after the configured horizon."""


class ReplayMismatchError(MarketError):
    """Raised when replay does not reproduce an expected state."""


class IdempotencyConflictError(MarketError):
    """Raised when an idempotency key is reused with a different payload."""


class StateVersionConflictError(MarketError):
    """Raised when an action references a stale or future state version."""
