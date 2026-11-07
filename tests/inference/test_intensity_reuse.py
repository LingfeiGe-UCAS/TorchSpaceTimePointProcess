from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from torchstpp.inference import update_background_probabilities, update_background_probabilities_distributed
from torchstpp.models.backgrounds import KDEState
from torchstpp.finite_source.em import finite_source_patch_responsibilities, update_finite_source_weights
from torchstpp.likelihood import ExactLikelihood, DistributedExactLikelihood
from tests.inference.test_finite_source_em import _problem


@pytest.mark.parametrize('ranges', [None, ((0, 1), (2, 3)), ()])
def test_reuses_intensity_with_different_target_chunks(ranges):
    model, catalog, state = _problem()
    reference_phi = update_background_probabilities(model, catalog, state, target_chunk_size=1, target_ranges=ranges)
    assert isinstance(reference_phi, KDEState)
    reference = finite_source_patch_responsibilities(model, catalog, state, target_chunk_size=1, target_ranges=ranges)
    result = update_background_probabilities(model, catalog, state, target_chunk_size=2,
                                            target_ranges=ranges, return_intensity=True)
    torch.testing.assert_close(result.background_state.weights, reference_phi.weights)
    with patch('torchstpp.finite_source.em._event_intensity', side_effect=AssertionError('recomputed')):
        actual = finite_source_patch_responsibilities(
            model, catalog, state, target_chunk_size=1, target_ranges=ranges,
            event_intensity=result.event_intensity,
        )
    torch.testing.assert_close(actual, reference)


@pytest.mark.parametrize('mutation', ['state', 'tau', 'catalog', 'identity', 'coverage'])
def test_rejects_stale_or_incompatible_intensities(mutation):
    model, catalog, state = _problem()
    result = update_background_probabilities(model, catalog, state, return_intensity=True,
                                            target_ranges=((0, 1),) if mutation == 'coverage' else None)
    if mutation == 'state':
        state.weights.mul_(0.5)
    elif mutation == 'tau':
        model.patch_tau.mul_(2)
    elif mutation == 'catalog':
        catalog.time.add_(0.1)
    elif mutation == 'identity':
        state = replace(state)
    with pytest.raises(ValueError):
        finite_source_patch_responsibilities(model, catalog, state, event_intensity=result.event_intensity)


def test_rejects_overlapping_producer_ranges():
    model, catalog, state = _problem()
    with pytest.raises(ValueError, match='disjoint'):
        update_background_probabilities(model, catalog, state, target_ranges=((0, 2), (1, 3)), return_intensity=True)


def _worker(rank, rendezvous, output, mode):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2)
    try:
        model, catalog, state = _problem()
        likelihood = DistributedExactLikelihood(
            ExactLikelihood(target_chunk_size=1, source_chunk_size=2),
            pair_budget=2 if mode == 'pair_budget' else None, source_alignment=1,
        )
        ranges = (((0, 3),) if rank == 0 else ()) if mode == 'empty_rank' else None
        sums = []
        original_sum = likelihood.global_sum_
        def record_sum(value):
            sums.append(tuple(value.shape))
            original_sum(value)
        with patch.object(likelihood, 'global_sum_', side_effect=record_sum):
            result = update_background_probabilities_distributed(
                model, catalog, state, likelihood, return_intensity=True, target_ranges=ranges,
            )
            with patch('torchstpp.finite_source.em._event_intensity', side_effect=AssertionError('recomputed')):
                tau = update_finite_source_weights(
                    model, catalog, state, distributed_likelihood=likelihood,
                    target_ranges=ranges, event_intensity=result.event_intensity,
                )
        torch.save(dict(phi=result.background_state.weights, tau=tau.patch_tau,
                        expected=tau.expected_patch_offspring, sums=sums),
                   Path(output) / f'rank{rank}.pt')
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('mode', ['block_cyclic', 'pair_budget', 'empty_rank'])
def test_two_ranks_reuse_local_intensity_without_extra_reduction(tmp_path, mode):
    if not dist.is_gloo_available():
        pytest.skip('Gloo unavailable')
    model, catalog, state = _problem()
    phi = update_background_probabilities(model, catalog, state)
    tau = update_finite_source_weights(model, catalog, state)
    mp.spawn(_worker, args=((tmp_path/'rendezvous').resolve().as_uri(), str(tmp_path), mode),
             nprocs=2, join=True)
    for rank in range(2):
        result = torch.load(tmp_path/f'rank{rank}.pt', weights_only=True)
        torch.testing.assert_close(result['phi'], phi.weights)
        torch.testing.assert_close(result['tau'], tau.patch_tau)
        torch.testing.assert_close(result['expected'], tau.expected_patch_offspring)
        assert result['sums'] == [(3,), (2,)]  # phi and patch responsibilities only
