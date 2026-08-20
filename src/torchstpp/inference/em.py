from collections.abc import Iterable
from typing import Any

import torch

from ..models.backgrounds import KDEState


@torch.no_grad()
def update_background_probabilities(
    model: Any,
    catalog: Any,
    state: KDEState,
    *,
    target_chunk_size: int = 2048,
    source_chunk_size: int = 4096,
) -> KDEState:
    """Perform one legacy ETAS E step and return the updated KDE state.

    ``state.weights`` stores phi_i, the conditional probability that event i
    belongs to the background process.  The model's scalar ``mu`` remains an
    MLE parameter and is not replaced by this event-level state.
    """
    state.validate()
    if target_chunk_size < 1 or source_chunk_size < 1:
        raise ValueError("chunk sizes must be positive")
    required = ("time", "position", "magnitude", "completeness_magnitude", "take")
    missing = [name for name in required if not hasattr(catalog, name)]
    if missing:
        raise TypeError(f"catalog is missing required attributes: {missing}")
    event_count = catalog.time.shape[0]
    if state.position.shape[0] != event_count:
        raise ValueError("KDEState and catalog must contain the same events")

    probability_blocks = []
    for target_begin in range(0, event_count, target_chunk_size):
        target_end = min(target_begin + target_chunk_size, event_count)
        target = catalog.take(slice(target_begin, target_end))
        background = model.background_rate(model.background.evaluate(target, state))
        intensity = background.clone()

        for source_begin in range(0, event_count, source_chunk_size):
            source_end = min(source_begin + source_chunk_size, event_count)
            source = catalog.take(slice(source_begin, source_end))
            if source.time[0] >= target.time[-1]:
                break
            intensity.add_(model.pairwise_rate(target, source).sum(dim=-1))

        if not torch.all(torch.isfinite(intensity)) or torch.any(intensity <= 0):
            raise FloatingPointError("event intensity must be finite and positive")
        probability_blocks.append((background / intensity).clamp_(0.0, 1.0))

    return KDEState(
        position=state.position,
        bandwidth=state.bandwidth,
        weights=torch.cat(probability_blocks),
        duration=state.duration,
        depth=state.depth,
    )


@torch.no_grad()
def expected_source_offspring(
    model: Any,
    catalog: Any,
    state: KDEState,
    *,
    target_chunk_size: int = 2048,
    source_chunk_size: int = 4096,
    target_ranges: Iterable[tuple[int, int]] | None = None,
) -> torch.Tensor:
    """Return ``sum_i P(i triggered by j)`` for every source event ``j``."""
    state.validate()
    if target_chunk_size < 1 or source_chunk_size < 1:
        raise ValueError("chunk sizes must be positive")
    event_count = catalog.time.shape[0]
    expected = catalog.time.new_zeros(event_count)
    ranges = target_ranges if target_ranges is not None else ((0, event_count),)
    for range_begin, range_end in ranges:
        if range_begin < 0 or range_end < range_begin or range_end > event_count:
            raise ValueError("target ranges must lie inside the catalog")
        for target_begin in range(range_begin, range_end, target_chunk_size):
            target_end = min(target_begin + target_chunk_size, range_end)
            target = catalog.take(slice(target_begin, target_end))
            background = model.background_rate(model.background.evaluate(target, state))
            intensity = background.clone()
            active_sources: list[tuple[int, int]] = []
            for source_begin in range(0, event_count, source_chunk_size):
                source_end = min(source_begin + source_chunk_size, event_count)
                source = catalog.take(slice(source_begin, source_end))
                if source.time[0] >= target.time[-1]:
                    break
                intensity.add_(model.pairwise_rate(target, source).sum(dim=-1))
                active_sources.append((source_begin, source_end))
            if not torch.all(torch.isfinite(intensity)) or torch.any(intensity <= 0):
                raise FloatingPointError("event intensity must be finite and positive")
            for source_begin, source_end in active_sources:
                source = catalog.take(slice(source_begin, source_end))
                probability = model.pairwise_rate(target, source) / intensity[:, None]
                expected[source_begin:source_end].add_(probability.sum(dim=0))
    return expected
