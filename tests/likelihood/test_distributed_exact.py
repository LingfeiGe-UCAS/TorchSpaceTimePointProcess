from __future__ import annotations

from datetime import timedelta
from pathlib import Path
import tempfile
from unittest import SkipTest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.likelihood.test_exact import make_problem
from tests.model.test_etas3d import make_problem as make_problem_3d
from torchstpp.inference import (
    FortranEMLBFGS,
    LegacyEMLBFGS,
    LegacyEMLBFGSConfig,
    TriggerWeightedMagnitudeSelector,
)
from torchstpp.likelihood import (
    DistributedExactLikelihood,
    ExactLikelihood,
    block_cyclic_ranges,
)


def _gradients(model):
    return {
        name: parameter.grad.detach().cpu().clone()
        for name, parameter in model.named_parameters()
    }


def _reference():
    model, catalog, domain, state = make_problem()
    exact = ExactLikelihood(
        target_chunk_size=2, source_chunk_size=2, checkpoint_blocks=True
    )
    loss = exact(model, catalog, domain, state)
    loss.backward()
    return float(loss.item()), _gradients(model)


def _fitter():
    return FortranEMLBFGS(
        LegacyEMLBFGSConfig(
            outer_iterations=1,
            e_steps_per_outer=1,
            decrease_e_steps=False,
            lbfgs_max_iter=1,
            line_search_fn=None,
            target_chunk_size=2,
            source_chunk_size=2,
        )
    )


def _fit_reference():
    model, catalog, domain, state = make_problem()
    result = _fitter().fit(
        model,
        ExactLikelihood(target_chunk_size=2, source_chunk_size=2),
        catalog,
        domain,
        state,
    )
    return {
        "negative_log_likelihood": result.negative_log_likelihood,
        "parameters": {
            name: value.detach().cpu().clone()
            for name, value in model.parameter_values().items()
        },
        "weights": result.background_probability.detach().cpu().clone(),
    }


def _reference_3d():
    model, catalog, domain, state = make_problem_3d()
    exact = ExactLikelihood(
        target_chunk_size=2, source_chunk_size=2, checkpoint_blocks=True
    )
    loss = exact(model, catalog, domain, state)
    loss.backward()
    return float(loss.item()), _gradients(model)


def _gloo_worker(rank: int, init_method: str, result_directory: str):
    initialized = False
    try:
        torch.set_num_threads(1)
        dist.init_process_group(
            backend="gloo",
            init_method=init_method,
            rank=rank,
            world_size=2,
            timeout=timedelta(seconds=90),
        )
        initialized = True
        model, catalog, domain, state = make_problem()
        likelihood = DistributedExactLikelihood(
            ExactLikelihood(
                target_chunk_size=2,
                source_chunk_size=2,
                checkpoint_blocks=True,
            )
        )
        likelihood.validate_replicas_(model, catalog, domain, state)
        terms = likelihood.terms(model, catalog, domain, state)
        likelihood.backward_and_sum_gradients_(terms.backward_loss, model)
        pair_model, pair_catalog, pair_domain, pair_state = make_problem()
        pair_likelihood = DistributedExactLikelihood(
            ExactLikelihood(
                target_chunk_size=2,
                source_chunk_size=2,
                checkpoint_blocks=True,
            ),
            pair_budget=6,
            source_alignment=1,
        )
        pair_terms = pair_likelihood.terms(
            pair_model, pair_catalog, pair_domain, pair_state
        )
        pair_likelihood.backward_and_sum_gradients_(
            pair_terms.backward_loss, pair_model
        )
        reference_selection = TriggerWeightedMagnitudeSelector().select(
            pair_model,
            pair_likelihood,
            pair_catalog,
            pair_state,
            target_chunk_size=2,
            source_chunk_size=2,
        )
        pair_plan = pair_likelihood.last_pair_budget_plan
        assert pair_plan is not None
        fit_model, fit_catalog, fit_domain, fit_state = make_problem()
        fit_likelihood = DistributedExactLikelihood(
            ExactLikelihood(target_chunk_size=2, source_chunk_size=2)
        )
        fit_result = _fitter().fit(
            fit_model,
            fit_likelihood,
            fit_catalog,
            fit_domain,
            fit_state,
        )
        model_3d, catalog_3d, domain_3d, state_3d = make_problem_3d()
        likelihood_3d = DistributedExactLikelihood(
            ExactLikelihood(
                target_chunk_size=2,
                source_chunk_size=2,
                checkpoint_blocks=True,
            )
        )
        likelihood_3d.validate_replicas_(
            model_3d, catalog_3d, domain_3d, state_3d
        )
        terms_3d = likelihood_3d.terms(
            model_3d, catalog_3d, domain_3d, state_3d
        )
        likelihood_3d.backward_and_sum_gradients_(
            terms_3d.backward_loss, model_3d
        )
        torch.save(
            {
                "loss": terms.global_terms.negative_log_likelihood.detach().cpu(),
                "target_count": terms.global_terms.target_count,
                "gradients": _gradients(model),
                "pair_budget": {
                    "loss": pair_terms.global_terms.negative_log_likelihood.detach().cpu(),
                    "gradients": _gradients(pair_model),
                    "target_ranges": pair_plan.target_ranges,
                    "source_ranges": pair_plan.source_ranges,
                    "target_chunk_size": pair_plan.target_chunk_size,
                    "source_chunk_size": pair_plan.source_chunk_size,
                    "reference_selection": reference_selection.as_dict(),
                },
                "fit": {
                    "negative_log_likelihood": fit_result.negative_log_likelihood,
                    "parameters": {
                        name: value.detach().cpu().clone()
                        for name, value in fit_model.parameter_values().items()
                    },
                    "weights": fit_result.background_probability.detach().cpu(),
                },
                "etas3d": {
                    "loss": terms_3d.global_terms.negative_log_likelihood.detach().cpu(),
                    "gradients": _gradients(model_3d),
                },
            },
            Path(result_directory) / f"rank-{rank}.pt",
        )
    finally:
        if initialized and dist.is_initialized():
            dist.destroy_process_group()


def test_block_cyclic_ranges_cover_work_once():
    rank0 = list(block_cyclic_ranges(7, 2, rank=0, world_size=2))
    rank1 = list(block_cyclic_ranges(7, 2, rank=1, world_size=2))
    assert rank0 == [(0, 2), (4, 6)]
    assert rank1 == [(2, 4), (6, 7)]


def test_single_process_wrapper_matches_exact_value_and_gradient():
    expected_loss, expected_gradients = _reference()
    model, catalog, domain, state = make_problem()
    likelihood = DistributedExactLikelihood(
        ExactLikelihood(
            target_chunk_size=2, source_chunk_size=2, checkpoint_blocks=True
        )
    )
    loss = likelihood(model, catalog, domain, state)
    likelihood.backward_and_sum_gradients_(loss, model)
    assert float(loss.item()) == expected_loss
    for name, actual in _gradients(model).items():
        assert torch.equal(actual, expected_gradients[name])


def test_single_process_pair_budget_matches_exact_value_and_gradient():
    expected_loss, expected_gradients = _reference()
    model, catalog, domain, state = make_problem()
    likelihood = DistributedExactLikelihood(
        ExactLikelihood(
            target_chunk_size=2, source_chunk_size=2, checkpoint_blocks=True
        ),
        pair_budget=6,
        source_alignment=1,
    )
    loss = likelihood(model, catalog, domain, state)
    likelihood.backward_and_sum_gradients_(loss, model)
    assert torch.allclose(
        loss.detach(), loss.new_tensor(expected_loss), rtol=1e-12, atol=1e-12
    )
    plan = likelihood.last_pair_budget_plan
    assert plan is not None
    assert plan.target_chunk_size * plan.source_chunk_size <= 6
    for name, actual in _gradients(model).items():
        assert torch.allclose(
            actual, expected_gradients[name], rtol=1e-9, atol=1e-10
        )


def test_two_process_gloo_matches_exact_value_and_gradient(tmp_path: Path):
    if not dist.is_available() or not dist.is_gloo_available():
        raise SkipTest("Gloo is unavailable")
    expected_loss, expected_gradients = _reference()
    expected_fit = _fit_reference()
    expected_3d_loss, expected_3d_gradients = _reference_3d()
    selection_model, selection_catalog, _, selection_state = make_problem()
    expected_selection = TriggerWeightedMagnitudeSelector().select(
        selection_model,
        ExactLikelihood(target_chunk_size=2, source_chunk_size=2),
        selection_catalog,
        selection_state,
        target_chunk_size=2,
        source_chunk_size=2,
    )
    rendezvous = (tmp_path / "rendezvous").resolve()
    result_directory = tmp_path / "results"
    result_directory.mkdir()
    mp.spawn(
        _gloo_worker,
        args=(rendezvous.as_uri(), str(result_directory)),
        nprocs=2,
        join=True,
    )
    for rank in range(2):
        result = torch.load(
            result_directory / f"rank-{rank}.pt", weights_only=True
        )
        assert result["target_count"] == 3
        assert torch.allclose(
            result["loss"], torch.tensor(expected_loss, dtype=result["loss"].dtype)
        )
        for name, actual in result["gradients"].items():
            assert torch.allclose(actual, expected_gradients[name], rtol=1e-9, atol=1e-10)
        assert torch.allclose(
            result["pair_budget"]["loss"],
            torch.tensor(
                expected_loss,
                dtype=result["pair_budget"]["loss"].dtype,
            ),
            rtol=1e-10,
            atol=1e-10,
        )
        assert result["pair_budget"]["target_chunk_size"] * result[
            "pair_budget"
        ]["source_chunk_size"] <= 6
        actual_selection = result["pair_budget"]["reference_selection"]
        assert actual_selection["selected_offset"] == expected_selection.selected_offset
        assert torch.allclose(
            torch.tensor(actual_selection["raw_weighted_magnitude"]),
            torch.tensor(expected_selection.raw_weighted_magnitude),
            rtol=1e-10,
            atol=1e-10,
        )
        target_indices = [
            index
            for begin, end in result["pair_budget"]["target_ranges"]
            for index in range(begin, end)
        ]
        source_indices = [
            index
            for begin, end in result["pair_budget"]["source_ranges"]
            for index in range(begin, end)
        ]
        assert target_indices == list(range(3))
        assert source_indices == list(range(6))
        for name, actual in result["pair_budget"]["gradients"].items():
            assert torch.allclose(
                actual, expected_gradients[name], rtol=1e-9, atol=1e-10
            )
        assert torch.allclose(
            torch.tensor(result["fit"]["negative_log_likelihood"]),
            torch.tensor(expected_fit["negative_log_likelihood"]),
            rtol=1e-10,
            atol=1e-10,
        )
        assert torch.allclose(
            result["fit"]["weights"], expected_fit["weights"],
            rtol=1e-10, atol=1e-10,
        )
        for name, actual in result["fit"]["parameters"].items():
            assert torch.allclose(
                actual, expected_fit["parameters"][name], rtol=1e-10, atol=1e-10
            )
        assert torch.allclose(
            result["etas3d"]["loss"],
            torch.tensor(
                expected_3d_loss, dtype=result["etas3d"]["loss"].dtype
            ),
            rtol=1e-10,
            atol=1e-10,
        )
        for name, actual in result["etas3d"]["gradients"].items():
            assert torch.allclose(
                actual, expected_3d_gradients[name], rtol=1e-9, atol=1e-10
            )


def test_distributed_wrapper_runs_current_inference_in_single_process():
    model, catalog, domain, state = make_problem()
    likelihood = DistributedExactLikelihood(
        ExactLikelihood(target_chunk_size=2, source_chunk_size=2)
    )
    fitter = LegacyEMLBFGS(
        LegacyEMLBFGSConfig(
            outer_iterations=1,
            e_steps_per_outer=1,
            decrease_e_steps=False,
            lbfgs_max_iter=1,
            line_search_fn=None,
            target_chunk_size=2,
            source_chunk_size=2,
        )
    )
    result = fitter.fit(model, likelihood, catalog, domain, state)
    assert result.outer_iterations == 1
    assert torch.isfinite(torch.tensor(result.negative_log_likelihood))


def _manual_main() -> None:
    """Run this module without requiring pytest in the runtime environment."""

    test_block_cyclic_ranges_cover_work_once()
    test_single_process_wrapper_matches_exact_value_and_gradient()
    test_single_process_pair_budget_matches_exact_value_and_gradient()
    with tempfile.TemporaryDirectory() as directory:
        test_two_process_gloo_matches_exact_value_and_gradient(Path(directory))
    test_distributed_wrapper_runs_current_inference_in_single_process()
    print("distributed exact tests: 5 passed", flush=True)


if __name__ == "__main__":
    _manual_main()
