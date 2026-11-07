from dataclasses import dataclass
from typing import Any

import torch

from ..models.backgrounds import KDEState


@dataclass(frozen=True)
class EventState:
    """Post-fit event quantities shared by output and relation extraction."""

    background_density: torch.Tensor
    background_intensity: torch.Tensor
    total_intensity: torch.Tensor
    phi: torch.Tensor

    def validate(self, event_count: int | None = None) -> None:
        expected = self.total_intensity.shape
        if self.total_intensity.ndim != 1:
            raise ValueError("event-state tensors must have shape [N]")
        for name in ("background_density", "background_intensity", "phi"):
            if getattr(self, name).shape != expected:
                raise ValueError("all event-state tensors must have the same shape")
        if event_count is not None and expected != (event_count,):
            raise ValueError("EventState and catalog must contain the same events")
        if not torch.all(torch.isfinite(self.total_intensity)):
            raise FloatingPointError("total event intensity contains non-finite values")
        if torch.any(self.total_intensity <= 0):
            raise FloatingPointError("total event intensity must be positive")

    def to(self, *args, **kwargs) -> "EventState":
        return EventState(
            background_density=self.background_density.to(*args, **kwargs),
            background_intensity=self.background_intensity.to(*args, **kwargs),
            total_intensity=self.total_intensity.to(*args, **kwargs),
            phi=self.phi.to(*args, **kwargs),
        )


@torch.no_grad()
def compute_event_state(
    model: Any,
    catalog: Any,
    background_state: KDEState,
    *,
    target_chunk_size: int = 2048,
    source_chunk_size: int = 4096,
) -> EventState:
    """Compute post-fit event intensity once for reuse by all output tasks."""
    if target_chunk_size < 1 or source_chunk_size < 1:
        raise ValueError("chunk sizes must be positive")
    required = ("time", "position", "magnitude", "completeness_magnitude", "take")
    missing = [name for name in required if not hasattr(catalog, name)]
    if missing:
        raise TypeError(f"catalog is missing required attributes: {missing}")
    event_count = catalog.time.shape[0]
    if event_count == 0:
        raise ValueError("catalog must contain at least one event")
    if background_state.position.shape[0] != event_count:
        raise ValueError("KDEState and catalog must contain the same events")

    density_blocks = []
    background_blocks = []
    intensity_blocks = []
    for target_begin in range(0, event_count, target_chunk_size):
        target_end = min(target_begin + target_chunk_size, event_count)
        target = catalog.take(slice(target_begin, target_end))
        density = model.background.evaluate(target, background_state)
        background = model.background_rate(density)
        intensity = background.clone()

        for source_begin in range(0, event_count, source_chunk_size):
            source_end = min(source_begin + source_chunk_size, event_count)
            source = catalog.take(slice(source_begin, source_end))
            if source.time[0] > target.time[-1]:
                break
            intensity.add_(model.pairwise_rate(target, source).sum(dim=-1))

        if not torch.all(torch.isfinite(intensity)) or torch.any(intensity <= 0):
            raise FloatingPointError("event intensity must be finite and positive")
        density_blocks.append(density)
        background_blocks.append(background)
        intensity_blocks.append(intensity)

    background_density = torch.cat(density_blocks)
    background_intensity = torch.cat(background_blocks)
    total_intensity = torch.cat(intensity_blocks)
    result = EventState(
        background_density=background_density,
        background_intensity=background_intensity,
        total_intensity=total_intensity,
        phi=(background_intensity / total_intensity).clamp(0.0, 1.0),
    )
    result.validate(event_count)
    return result
