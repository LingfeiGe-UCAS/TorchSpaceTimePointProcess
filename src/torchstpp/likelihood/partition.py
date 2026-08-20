"""Deterministic target partitioning for distributed exact likelihoods."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PairBudgetPlan:
    """One common distributed plan derived from a pair-element budget.

    ``target_ranges`` are disjoint, time-contiguous ranges into the selected
    target catalog. ``target_chunk_size`` and ``source_chunk_size`` are common
    compute tile sizes used by every rank.
    """

    target_ranges: tuple[tuple[int, int], ...]
    source_ranges: tuple[tuple[int, int], ...]
    target_chunk_size: int
    source_chunk_size: int
    pair_budget: int

    def target_range(self, rank: int) -> tuple[int, int]:
        return self.target_ranges[rank]

    def source_range(self, rank: int) -> tuple[int, int]:
        return self.source_ranges[rank]


def even_contiguous_ranges(
    total: int,
    *,
    world_size: int,
) -> tuple[tuple[int, int], ...]:
    """Split ``range(total)`` into disjoint contiguous count-balanced ranges."""

    if total < 0:
        raise ValueError("total must be non-negative")
    if world_size < 1:
        raise ValueError("world_size must be positive")
    return tuple(
        (
            total * rank // world_size,
            total * (rank + 1) // world_size,
        )
        for rank in range(world_size)
    )


def causal_work_ranges(
    target_time: torch.Tensor,
    source_time: torch.Tensor,
    *,
    world_size: int,
    background_source_count: int,
) -> tuple[tuple[int, int], ...]:
    """Split sorted targets into time-contiguous ranges of similar pair work.

    The estimated cost of target ``i`` is the number of KDE background
    centers plus the number of strictly earlier triggering sources. This is a
    deterministic scheduling estimate; it does not change the ETAS formula.
    """

    if world_size < 1:
        raise ValueError("world_size must be positive")
    if background_source_count < 0:
        raise ValueError("background_source_count must be non-negative")
    if target_time.ndim != 1 or source_time.ndim != 1:
        raise ValueError("target_time and source_time must be one-dimensional")
    if target_time.numel() == 0:
        return tuple((0, 0) for _ in range(world_size))
    if target_time.numel() > 1 and bool(
        torch.any(target_time[1:] < target_time[:-1])
    ):
        raise ValueError("target_time must be sorted by non-decreasing time")
    if source_time.numel() > 1 and bool(
        torch.any(source_time[1:] < source_time[:-1])
    ):
        raise ValueError("source_time must be sorted by non-decreasing time")

    earlier = torch.searchsorted(
        source_time.contiguous(),
        target_time.contiguous(),
        right=False,
    ).to(dtype=torch.int64)
    work = earlier + int(background_source_count)
    # A zero-cost event is still real work that must belong to exactly one rank.
    work = torch.clamp(work, min=1)
    cumulative = torch.cumsum(work, dim=0)
    total_work = int(cumulative[-1].item())
    target_count = int(target_time.numel())

    boundaries = [0]
    for rank in range(1, world_size):
        threshold = (total_work * rank + world_size - 1) // world_size
        needle = cumulative.new_tensor(threshold)
        boundary = int(torch.searchsorted(cumulative, needle).item()) + 1
        boundaries.append(max(boundaries[-1], min(boundary, target_count)))
    boundaries.append(target_count)
    return tuple(
        (boundaries[rank], boundaries[rank + 1])
        for rank in range(world_size)
    )


def derive_pair_chunks(
    *,
    pair_budget: int,
    target_ranges: tuple[tuple[int, int], ...],
    target_chunk_cap: int,
    source_count: int,
    source_alignment: int = 256,
) -> tuple[int, int]:
    """Derive common target/source compute chunks with ``s * t <= budget``."""

    if pair_budget < 1:
        raise ValueError("pair_budget must be positive")
    if target_chunk_cap < 1:
        raise ValueError("target_chunk_cap must be positive")
    if source_count < 1:
        raise ValueError("source_count must be positive")
    if source_alignment < 1:
        raise ValueError("source_alignment must be positive")
    if not target_ranges:
        raise ValueError("target_ranges must not be empty")

    largest_local_target_count = max(end - begin for begin, end in target_ranges)
    if largest_local_target_count < 1:
        raise ValueError("at least one rank must own a target event")
    target_chunk_size = min(target_chunk_cap, largest_local_target_count)
    raw_source_chunk_size = pair_budget // target_chunk_size
    if raw_source_chunk_size < 1:
        raise ValueError(
            "pair_budget is smaller than the required target compute chunk"
        )

    source_chunk_size = min(source_count, raw_source_chunk_size)
    if source_chunk_size >= source_alignment:
        source_chunk_size = (
            source_chunk_size // source_alignment
        ) * source_alignment
    source_chunk_size = max(1, source_chunk_size)
    return target_chunk_size, source_chunk_size


def build_pair_budget_plan(
    target_time: torch.Tensor,
    source_time: torch.Tensor,
    *,
    world_size: int,
    pair_budget: int,
    target_chunk_cap: int,
    background_source_count: int,
    source_alignment: int = 256,
) -> PairBudgetPlan:
    """Build one deterministic plan shared by every distributed rank."""

    target_ranges = causal_work_ranges(
        target_time,
        source_time,
        world_size=world_size,
        background_source_count=background_source_count,
    )
    source_ranges = even_contiguous_ranges(
        int(source_time.numel()),
        world_size=world_size,
    )
    target_chunk_size, source_chunk_size = derive_pair_chunks(
        pair_budget=pair_budget,
        target_ranges=target_ranges,
        target_chunk_cap=target_chunk_cap,
        source_count=int(source_time.numel()),
        source_alignment=source_alignment,
    )
    return PairBudgetPlan(
        target_ranges=target_ranges,
        source_ranges=source_ranges,
        target_chunk_size=target_chunk_size,
        source_chunk_size=source_chunk_size,
        pair_budget=pair_budget,
    )
