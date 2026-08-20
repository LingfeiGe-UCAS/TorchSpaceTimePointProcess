import torch

from torchstpp.likelihood import (
    build_pair_budget_plan,
    causal_work_ranges,
    derive_pair_chunks,
    even_contiguous_ranges,
)


def test_even_ranges_cover_every_index_once():
    ranges = even_contiguous_ranges(11, world_size=4)
    assert ranges == ((0, 2), (2, 5), (5, 8), (8, 11))
    indices = [index for begin, end in ranges for index in range(begin, end)]
    assert indices == list(range(11))


def test_causal_work_ranges_give_later_targets_fewer_events():
    time = torch.arange(12, dtype=torch.float64)
    ranges = causal_work_ranges(
        time,
        time,
        world_size=3,
        background_source_count=1,
    )
    assert ranges == ((0, 7), (7, 10), (10, 12))
    indices = [index for begin, end in ranges for index in range(begin, end)]
    assert indices == list(range(12))


def test_pair_chunks_respect_budget_cap_and_alignment():
    target_chunk, source_chunk = derive_pair_chunks(
        pair_budget=64,
        target_ranges=((0, 7), (7, 10), (10, 12)),
        target_chunk_cap=4,
        source_count=100,
        source_alignment=4,
    )
    assert target_chunk == 4
    assert source_chunk == 16
    assert target_chunk * source_chunk <= 64


def test_build_pair_budget_plan_separates_scheduling_from_compute_chunks():
    time = torch.arange(12, dtype=torch.float64)
    plan = build_pair_budget_plan(
        time,
        time,
        world_size=3,
        pair_budget=16,
        target_chunk_cap=4,
        background_source_count=1,
        source_alignment=1,
    )
    assert plan.target_ranges == ((0, 7), (7, 10), (10, 12))
    assert plan.source_ranges == ((0, 4), (4, 8), (8, 12))
    assert plan.target_chunk_size == 4
    assert plan.source_chunk_size == 4
    assert plan.target_chunk_size * plan.source_chunk_size <= plan.pair_budget
