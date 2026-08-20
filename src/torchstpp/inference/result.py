from dataclasses import dataclass

import torch

from ..models.backgrounds import KDEState


@dataclass(frozen=True)
class OptimizationStep:
    """Diagnostics recorded after one outer EM--MLE iteration."""

    iteration: int
    negative_log_likelihood: float
    background_change: float
    parameter_change: float
    parameter_values: dict[str, float]


@dataclass(frozen=True)
class FitResult:
    """Result of one optimization run."""

    negative_log_likelihood: float
    background_state: KDEState
    history: tuple[OptimizationStep, ...]
    converged: bool
    outer_iterations: int
    stop_reason: str | None = None

    @property
    def background_probability(self) -> torch.Tensor:
        """Per-event background probabilities, traditionally named ``phi``."""
        return self.background_state.weights
