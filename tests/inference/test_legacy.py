import torch

from torchstpp.inference import (
    FortranEMLBFGS,
    LegacyEMLBFGS,
    LegacyEMLBFGSConfig,
    update_background_probabilities,
)

from tests.likelihood.test_exact import make_problem


def test_e_step_updates_phi_without_changing_model_parameters():
    model, catalog, _, state = make_problem()
    before = {name: value.detach().clone() for name, value in model.parameter_values().items()}
    updated = update_background_probabilities(
        model, catalog, state, target_chunk_size=2, source_chunk_size=2
    )

    assert updated.weights.shape == catalog.time.shape
    assert torch.all((updated.weights >= 0.0) & (updated.weights <= 1.0))
    assert not updated.weights.requires_grad
    for name, value in model.parameter_values().items():
        assert torch.equal(value, before[name])


def test_legacy_optimizer_runs_em_and_lbfgs():
    model, catalog, domain, state = make_problem()
    from torchstpp.likelihood import ExactLikelihood

    likelihood = ExactLikelihood(target_chunk_size=2, source_chunk_size=2)
    fitter = LegacyEMLBFGS(
        LegacyEMLBFGSConfig(
            outer_iterations=2,
            e_steps_per_outer=1,
            decrease_e_steps=False,
            background_tolerance=0.0,
            parameter_tolerance=0.0,
            lbfgs_max_iter=1,
            target_chunk_size=2,
            source_chunk_size=2,
        )
    )
    result = fitter.fit(model, likelihood, catalog, domain, state)

    assert result.outer_iterations == 2
    assert len(result.history) == 2
    assert result.history[-1].parameter_change >= 0.0
    assert torch.isfinite(torch.tensor(result.negative_log_likelihood))
    assert result.background_probability.shape == catalog.time.shape


def test_fortran_em_optimizes_with_kde_from_before_final_phi_update():
    model, catalog, domain, state = make_problem()
    from torchstpp.likelihood import ExactLikelihood

    exact = ExactLikelihood(target_chunk_size=2, source_chunk_size=2)
    likelihood_states = []

    def recording_likelihood(model, catalog, domain, background_state):
        likelihood_states.append(background_state.weights.detach().clone())
        return exact(model, catalog, domain, background_state)

    fitter = FortranEMLBFGS(
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
    result = fitter.fit(model, recording_likelihood, catalog, domain, state)

    assert likelihood_states
    assert all(torch.equal(weights, state.weights) for weights in likelihood_states)
    assert not torch.equal(result.background_probability, state.weights)


def test_lbfgs_curvature_history_is_reset_for_each_outer_iteration():
    model, catalog, domain, state = make_problem()
    from torchstpp.likelihood import ExactLikelihood

    class CountingFitter(LegacyEMLBFGS):
        optimizer_count = 0

        def _new_optimizer(self, trainable):
            self.optimizer_count += 1
            return super()._new_optimizer(trainable)

    fitter = CountingFitter(
        LegacyEMLBFGSConfig(
            outer_iterations=3,
            e_steps_per_outer=1,
            decrease_e_steps=False,
            background_tolerance=0.0,
            parameter_tolerance=0.0,
            lbfgs_max_iter=1,
            line_search_fn=None,
            target_chunk_size=2,
            source_chunk_size=2,
        )
    )
    fitter.fit(
        model,
        ExactLikelihood(target_chunk_size=2, source_chunk_size=2),
        catalog,
        domain,
        state,
    )

    assert fitter.optimizer_count == 3


def test_degenerate_monitor_stops_after_recording_the_iteration():
    model, catalog, domain, state = make_problem()
    from torchstpp.likelihood import ExactLikelihood

    fitter = LegacyEMLBFGS(
        LegacyEMLBFGSConfig(
            outer_iterations=3,
            e_steps_per_outer=1,
            decrease_e_steps=False,
            lbfgs_max_iter=1,
            line_search_fn=None,
            target_chunk_size=2,
            source_chunk_size=2,
            stop_on_degenerate=True,
            maximum_absolute_value=0.1,
        )
    )
    result = fitter.fit(
        model,
        ExactLikelihood(target_chunk_size=2, source_chunk_size=2),
        catalog,
        domain,
        state,
    )

    assert result.outer_iterations == 1
    assert len(result.history) == 1
    assert result.stop_reason is not None
    assert "reached |value|" in result.stop_reason
