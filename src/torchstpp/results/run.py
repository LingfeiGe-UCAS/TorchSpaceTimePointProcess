from dataclasses import dataclass
from typing import Any

from ..inference import FitResult
from ..models.backgrounds import KDEState
from .event_state import EventState


@dataclass
class FittedRun:
    """In-memory fitted state reused by output and post-fit analyses."""

    model: Any
    catalog: Any
    domain: Any
    background_state: KDEState
    event_state: EventState
    likelihood: Any
    fit_result: FitResult
    config: dict[str, Any]
    elapsed_seconds: float | None = None
    fit_elapsed_seconds: float | None = None
    peak_memory_bytes: int | None = None

    def __post_init__(self) -> None:
        if not hasattr(self.catalog, "time"):
            raise TypeError("catalog must provide time")
        self.event_state.validate(self.catalog.time.shape[0])
