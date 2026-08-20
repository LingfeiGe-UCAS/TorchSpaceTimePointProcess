import torch

from tests.model.test_etas3d import make_problem
from torchstpp.config import build_inference, build_likelihood, load_config
from torchstpp.inference import (
    AdaptiveReferenceInference,
    FortranEMLBFGS,
    LegacyEMLBFGSConfig,
    TriggerWeightedMagnitudeSelector,
    expected_source_offspring,
)
from torchstpp.likelihood import DistributedExactLikelihood, ExactLikelihood


def test_disjoint_target_ranges_sum_to_single_catalog_result():
    model, catalog, _, state = make_problem()
    expected = expected_source_offspring(
        model, catalog, state, target_chunk_size=2, source_chunk_size=2
    )
    first = expected_source_offspring(
        model,
        catalog,
        state,
        target_chunk_size=2,
        source_chunk_size=2,
        target_ranges=((0, 2),),
    )
    second = expected_source_offspring(
        model,
        catalog,
        state,
        target_chunk_size=2,
        source_chunk_size=2,
        target_ranges=((2, len(catalog.time)),),
    )
    assert torch.allclose(first + second, expected)


def test_distributed_wrapper_exposes_active_target_schedule_in_single_process():
    likelihood = DistributedExactLikelihood(
        ExactLikelihood(target_chunk_size=2, source_chunk_size=3), pair_budget=6
    )
    time = torch.arange(5, dtype=torch.float64)
    ranges = likelihood.local_target_ranges(
        time, time, background_source_count=5
    )
    covered = [index for begin, end in ranges for index in range(begin, end)]
    assert covered == list(range(5))


def test_auto_reference_config_builds_a_composable_wrapper():
    config = load_config(
        "configs/qtm_mc2_relocated_etas3d_float64_auto_mref_min3estep_11em_checkpoint.toml"
    )
    likelihood = build_likelihood(config)
    inference = build_inference(config, likelihood=likelihood)
    assert isinstance(inference, AdaptiveReferenceInference)
    assert isinstance(inference.selector, TriggerWeightedMagnitudeSelector)
    assert inference.pilot.config.outer_iterations == 3
    assert inference.base.config.outer_iterations == 11


def test_adaptive_wrapper_runs_pilot_selection_and_main_fit():
    model, catalog, domain, state = make_problem()
    optimizer_config = LegacyEMLBFGSConfig(
        outer_iterations=1,
        e_steps_per_outer=1,
        decrease_e_steps=False,
        lbfgs_max_iter=1,
        line_search_fn=None,
        target_chunk_size=2,
        source_chunk_size=2,
    )
    inference = AdaptiveReferenceInference(
        pilot=FortranEMLBFGS(optimizer_config),
        base=FortranEMLBFGS(optimizer_config),
        selector=TriggerWeightedMagnitudeSelector(
            minimum_offset=0.1, maximum_offset=1.0, round_to=0.1
        ),
        pilot_outer_iterations=1,
    )
    result = inference.fit(
        model,
        ExactLikelihood(target_chunk_size=2, source_chunk_size=2),
        catalog,
        domain,
        state,
    )
    assert result.outer_iterations == 1
    assert inference.last_selection is not None
    assert model.depth_reference_selection["pilot_outer_iterations"] == 1
    assert abs(
        model.depth_reference_magnitude_offset
        - inference.last_selection.selected_offset
    ) < 1e-6
