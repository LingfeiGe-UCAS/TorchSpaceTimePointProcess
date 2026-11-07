from unittest.mock import patch

import pytest
import torch

from torchstpp.inference import FortranEMLBFGS, LegacyEMLBFGSConfig
from torchstpp.likelihood import ExactLikelihood, DistributedExactLikelihood
from tests.likelihood.test_exact import make_problem
from tests.model.test_etas3d import make_problem as make_problem_3d


@pytest.mark.parametrize("factory", [make_problem, make_problem_3d])
@pytest.mark.parametrize("checkpoint", [False, True])
@pytest.mark.parametrize("layout", ["single", "block_cyclic", "pair_budget"])
def test_repeated_loss_and_gradient_reuse_without_kde_calls(factory, checkpoint, layout):
    model, catalog, domain, state = factory()
    exact = ExactLikelihood(target_chunk_size=2, source_chunk_size=2, checkpoint_blocks=checkpoint)
    likelihood = exact if layout == "single" else DistributedExactLikelihood(
        exact, pair_budget=4 if layout == "pair_budget" else None, source_alignment=1,
    )
    with patch.object(model.background, 'integrate', wraps=model.background.integrate) as integrate:
        prepared = likelihood.prepare_m_step(model, catalog, domain, state)
        assert integrate.call_count == 1
    parameters = [p for p in model.parameters() if p.requires_grad]
    for _ in range(2):
        # Trainable parameters, including mu, change within a fixed-background M step.
        with torch.no_grad():
            parameters[0].add_(0.01)
        reference = likelihood(model, catalog, domain, state)
        grad_reference = torch.autograd.grad(reference, parameters)
        with patch.object(model.background, 'evaluate', side_effect=AssertionError('recomputed KDE')), \
             patch.object(model.background, 'integrate', side_effect=AssertionError('recomputed integral')):
            result = prepared(model, catalog, domain, state)
            grad_result = torch.autograd.grad(result, parameters)
        torch.testing.assert_close(result, reference, rtol=1e-12, atol=1e-12)
        for a, b in zip(grad_result, grad_reference):
            torch.testing.assert_close(a, b, rtol=1e-11, atol=1e-11)
    assert not hasattr(catalog, '_m_step_background_values')
    assert not hasattr(state, '_m_step_background_integral')


def test_new_state_requires_new_preparation():
    from dataclasses import replace
    model, catalog, domain, state = make_problem()
    likelihood = ExactLikelihood(target_chunk_size=2, source_chunk_size=2)
    prepared = likelihood.prepare_m_step(model, catalog, domain, state)
    updated = replace(state, weights=state.weights * 0.5)
    with pytest.raises(ValueError, match='another fitting state'):
        prepared(model, catalog, domain, updated)
    new_prepared = likelihood.prepare_m_step(model, catalog, domain, updated)
    torch.testing.assert_close(new_prepared(model, catalog, domain, updated),
                               likelihood(model, catalog, domain, updated))


def test_trainable_background_inputs_are_not_detached():
    model, catalog, domain, state = make_problem()
    state.weights.requires_grad_(True)
    exact = ExactLikelihood(target_chunk_size=2, source_chunk_size=2)
    assert exact.prepare_m_step(model, catalog, domain, state) is exact
    grad = torch.autograd.grad(exact(model, catalog, domain, state), state.weights)[0]
    assert torch.isfinite(grad).all()


def test_polygon_integrator_keeps_concrete_source_fields():
    from torchstpp.integration import PolygonSpatialIntegrator
    model, catalog, domain, state = make_problem()
    model.spatial_integrator = PolygonSpatialIntegrator(subdivisions=5, source_chunk_size=2)
    exact = ExactLikelihood(target_chunk_size=2, source_chunk_size=2, checkpoint_blocks=True)
    prepared = exact.prepare_m_step(model, catalog, domain, state)
    parameters = list(model.parameters())
    reference = exact(model, catalog, domain, state)
    reference_grads = torch.autograd.grad(reference, parameters)
    actual = prepared(model, catalog, domain, state)
    actual_grads = torch.autograd.grad(actual, parameters)
    torch.testing.assert_close(actual, reference)
    for a, b in zip(actual_grads, reference_grads):
        torch.testing.assert_close(a, b)


def test_two_fortran_outer_steps_match_uncached_and_prepare_lagged_state():
    model, catalog, domain, state = make_problem()
    reference_model, reference_catalog, reference_domain, reference_state = make_problem()
    exact = ExactLikelihood(target_chunk_size=2, source_chunk_size=2)
    config = LegacyEMLBFGSConfig(
        outer_iterations=2, e_steps_per_outer=1, decrease_e_steps=False,
        lbfgs_max_iter=2, line_search_fn=None, target_chunk_size=2, source_chunk_size=2,
    )
    prepare = exact.prepare_m_step
    prepared_weights = []
    def record(model, catalog, domain, background_state):
        prepared_weights.append(background_state.weights.clone())
        return prepare(model, catalog, domain, background_state)
    with patch.object(exact, 'prepare_m_step', side_effect=record):
        result = FortranEMLBFGS(config).fit(model, exact, catalog, domain, state)
    assert len(prepared_weights) == 2
    torch.testing.assert_close(prepared_weights[0], state.weights)
    assert not torch.equal(prepared_weights[1], prepared_weights[0])
    reference = FortranEMLBFGS(config).fit(
        reference_model, lambda *args: exact(*args), reference_catalog,
        reference_domain, reference_state,
    )
    for a, b in zip(result.history, reference.history):
        assert a.negative_log_likelihood == pytest.approx(b.negative_log_likelihood, abs=1e-10)
        for key in a.parameter_values:
            assert a.parameter_values[key] == pytest.approx(b.parameter_values[key], rel=1e-10)
    torch.testing.assert_close(result.background_probability, reference.background_probability)
