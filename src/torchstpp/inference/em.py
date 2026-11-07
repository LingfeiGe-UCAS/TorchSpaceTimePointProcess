from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch

from ..models.backgrounds import KDEState
from ..evaluation import EventIntensity


@dataclass(frozen=True)
class BackgroundProbabilityUpdate:
    """Updated phi plus total intensity evaluated using the OLD E-step state."""
    background_state: KDEState
    event_intensity: EventIntensity


@torch.no_grad()
def update_background_probabilities(
    model: Any,
    catalog: Any,
    state: KDEState,
    *,
    target_chunk_size: int = 2048,
    source_chunk_size: int = 4096,
    target_ranges: Iterable[tuple[int, int]] | None = None,
    return_intensity: bool = False,
) -> KDEState | BackgroundProbabilityUpdate:
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

    ranges = tuple(target_ranges) if target_ranges is not None else ((0, event_count),)
    occupied = sorted((begin, end) for begin, end in ranges if end > begin)
    if any(right[0] < left[1] for left, right in zip(occupied, occupied[1:])):
        raise ValueError('target ranges must be disjoint')
    weights = torch.zeros_like(state.weights)
    intensity_values = torch.full_like(catalog.time, float('nan')) if return_intensity else None
    for range_begin, range_end in ranges:
        if range_begin < 0 or range_end < range_begin or range_end > event_count:
            raise ValueError("target ranges must lie inside the catalog")
        for target_begin in range(range_begin, range_end, target_chunk_size):
            target_end = min(target_begin + target_chunk_size, range_end)
            target = catalog.take(slice(target_begin, target_end))
            background = model.background_rate(model.background.evaluate(target, state))
            intensity = background.clone()

            for source_begin in range(0, event_count, source_chunk_size):
                source_end = min(source_begin + source_chunk_size, event_count)
                source = catalog.take(slice(source_begin, source_end))
                if source.time[0] > target.time[-1]:
                    break
                intensity.add_(model.pairwise_rate(target, source).sum(dim=-1))

            if not torch.all(torch.isfinite(intensity)) or torch.any(intensity <= 0):
                raise FloatingPointError("event intensity must be finite and positive")
            weights[target_begin:target_end] = (background / intensity).clamp_(
                0.0, 1.0
            )
            if intensity_values is not None:
                intensity_values[target_begin:target_end] = intensity

    updated = KDEState(
        position=state.position,
        bandwidth=state.bandwidth,
        weights=weights,
        duration=state.duration,
        depth=state.depth,
    )
    if intensity_values is not None:
        return BackgroundProbabilityUpdate(updated, EventIntensity(intensity_values, model, catalog, state))
    return updated


@torch.no_grad()
def update_background_probabilities_distributed(
    model: Any,
    catalog: Any,
    state: KDEState,
    distributed_likelihood: Any,
    *,
    target_chunk_size: int = 2048,
    source_chunk_size: int = 4096,
    target_ranges: Iterable[tuple[int, int]] | None = None,
    return_intensity: bool = False,
) -> KDEState | BackgroundProbabilityUpdate:
    """Perform one exact target-parallel E step on replicated source history.

    Every rank evaluates only its disjoint target ranges, while retaining the
    complete source catalog and KDE state needed by each conditional intensity.
    The local full-length weight vectors have zeros outside the owned ranges;
    one distributed SUM therefore reconstructs the same complete ``phi`` vector
    on every rank without communicating inside the quadratic pair loop.

    The object passed as ``distributed_likelihood`` must provide the public
    scheduling and collective helpers of ``DistributedExactLikelihood``.
    When its pair-budget scheduler is active, the E step uses the same derived
    target and source tile sizes as the distributed likelihood.
    """

    local_target_ranges = getattr(
        distributed_likelihood, "local_target_ranges", None
    )
    local_target_schedule = getattr(
        distributed_likelihood, "local_target_schedule", None
    )
    global_sum = getattr(distributed_likelihood, "global_sum_", None)
    synchronize_failure = getattr(
        distributed_likelihood, "synchronize_failure", None
    )
    if not callable(local_target_ranges) or not callable(global_sum):
        raise TypeError(
            "distributed_likelihood must provide local_target_ranges() "
            "and global_sum_()"
        )

    updated: KDEState | None = None
    evaluation: BackgroundProbabilityUpdate | None = None
    error: Exception | None = None
    try:
        if target_ranges is None:
            if callable(local_target_schedule):
                ranges, effective_target_chunk, effective_source_chunk = (
                    local_target_schedule(
                        catalog.time,
                        catalog.time,
                        background_source_count=int(state.position.shape[0]),
                    )
                )
            else:
                ranges = local_target_ranges(
                    catalog.time,
                    catalog.time,
                    background_source_count=int(state.position.shape[0]),
                )
                effective_target_chunk = target_chunk_size
                effective_source_chunk = source_chunk_size
        else:
            ranges = tuple(target_ranges)
            effective_target_chunk = target_chunk_size
            effective_source_chunk = source_chunk_size
        updated = update_background_probabilities(
            model,
            catalog,
            state,
            target_chunk_size=effective_target_chunk,
            source_chunk_size=effective_source_chunk,
            target_ranges=ranges,
            return_intensity=return_intensity,
        )
        if isinstance(updated, BackgroundProbabilityUpdate):
            evaluation = updated
            updated = evaluation.background_state
    except Exception as caught:
        error = caught

    if callable(synchronize_failure):
        synchronize_failure(error, catalog.time, stage="distributed E step")
    elif error is not None:
        raise error
    assert updated is not None

    global_sum(updated.weights)
    reduction_error: Exception | None = None
    try:
        if not bool(torch.all(torch.isfinite(updated.weights))):
            raise FloatingPointError(
                "distributed background probabilities must be finite"
            )
        if bool(torch.any((updated.weights < 0.0) | (updated.weights > 1.0))):
            raise FloatingPointError(
                "distributed background probabilities must lie in [0, 1]"
            )
    except Exception as caught:
        reduction_error = caught
    if callable(synchronize_failure):
        synchronize_failure(
            reduction_error,
            catalog.time,
            stage="distributed E-step reduction validation",
        )
    elif reduction_error is not None:
        raise reduction_error
    # Only phi is reduced. Each rank retains its local intensity rows for tau;
    # no additional all-reduce or all-gather of the intensity vector is needed.
    return evaluation if evaluation is not None else updated


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
                if source.time[0] > target.time[-1]:
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
