import torch

from torchstpp.inference import (
    FortranEMLBFGS,
    LegacyEMLBFGS,
    LegacyEMLBFGSConfig,
    update_background_probabilities,
    update_background_probabilities_distributed,
)
from torchstpp.likelihood import DistributedExactLikelihood, ExactLikelihood

from tests.likelihood.test_exact import make_problem
from tests.model.test_etas3d import make_problem as make_problem_3d


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


def test_disjoint_target_ranges_reconstruct_the_full_e_step():
    model, catalog, _, state = make_problem()
    expected = update_background_probabilities(
        model, catalog, state, target_chunk_size=2, source_chunk_size=2
    )
    rank_zero = update_background_probabilities(
        model,
        catalog,
        state,
        target_chunk_size=2,
        source_chunk_size=2,
        target_ranges=((0, 2), (4, 6)),
    )
    rank_one = update_background_probabilities(
        model,
        catalog,
        state,
        target_chunk_size=2,
        source_chunk_size=2,
        target_ranges=((2, 4),),
    )

    assert torch.equal(rank_zero.weights + rank_one.weights, expected.weights)


def test_single_process_distributed_e_step_honors_pair_budget():
    model, catalog, _, state = make_problem()
    expected = update_background_probabilities(
        model, catalog, state, target_chunk_size=2, source_chunk_size=2
    )
    likelihood = DistributedExactLikelihood(
        ExactLikelihood(target_chunk_size=2, source_chunk_size=2),
        pair_budget=4,
        source_alignment=1,
    )
    ranges, target_chunk, source_chunk = likelihood.local_target_schedule(
        catalog.time,
        catalog.time,
        background_source_count=state.position.shape[0],
    )

    actual = update_background_probabilities_distributed(
        model,
        catalog,
        state,
        likelihood,
        target_chunk_size=2,
        source_chunk_size=2,
    )

    assert ranges
    assert target_chunk * source_chunk <= 4
    assert likelihood.last_pair_budget_plan is None
    assert torch.allclose(actual.weights, expected.weights, rtol=1e-12, atol=1e-12)


def test_distributed_e_step_preserves_3d_depth_state():
    model, catalog, _, state = make_problem_3d()
    expected = update_background_probabilities(
        model, catalog, state, target_chunk_size=2, source_chunk_size=2
    )
    likelihood = DistributedExactLikelihood(
        ExactLikelihood(target_chunk_size=2, source_chunk_size=2)
    )

    actual = update_background_probabilities_distributed(
        model,
        catalog,
        state,
        likelihood,
        target_chunk_size=2,
        source_chunk_size=2,
    )

    assert actual.depth is state.depth
    assert torch.equal(actual.weights, expected.weights)


def test_legacy_optimizer_runs_em_and_lbfgs():
    model, catalog, domain, state = make_problem()

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


def test_lbfgs_max_iter_schedule_is_applied_per_outer_iteration():
    model, catalog, domain, state = make_problem()
    fitter = LegacyEMLBFGS(
        LegacyEMLBFGSConfig(
            outer_iterations=3,
            e_steps_per_outer=1,
            decrease_e_steps=False,
            background_tolerance=0.0,
            parameter_tolerance=0.0,
            lbfgs_max_iter=9,
            lbfgs_max_iter_schedule=(1, 2, 3),
            lbfgs_max_eval_schedule=(4, 5, 6),
            line_search_fn=None,
            target_chunk_size=2,
            source_chunk_size=2,
        )
    )

    result = fitter.fit(
        model,
        ExactLikelihood(target_chunk_size=2, source_chunk_size=2),
        catalog,
        domain,
        state,
    )

    assert [step.m_step_max_iter for step in result.history] == [1, 2, 3]
    assert [step.m_step_max_eval for step in result.history] == [4, 5, 6]
    assert all(step.m_step_iterations is not None for step in result.history)
    assert all(
        step.m_step_function_evaluations is not None for step in result.history
    )


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


def test_lbfgs_curvature_history_can_be_retained_across_outer_iterations():
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
            lbfgs_max_iter=2,
            lbfgs_history_size=5,
            retain_lbfgs_history_across_outer_iterations=True,
            line_search_fn=None,
            target_chunk_size=2,
            source_chunk_size=2,
        )
    )
    result = fitter.fit(
        model,
        ExactLikelihood(target_chunk_size=2, source_chunk_size=2),
        catalog,
        domain,
        state,
    )

    assert fitter.optimizer_count == 1
    assert len(result.history) == 3
    assert all(
        step.m_step_iterations is not None
        and 0 <= step.m_step_iterations <= 2
        for step in result.history
    )
    assert all(
        step.m_step_function_evaluations is not None
        and step.m_step_function_evaluations >= 1
        for step in result.history
    )


def test_deprecated_history_alias_still_reuses_one_optimizer():
    model, catalog, domain, state = make_problem()
    from torchstpp.likelihood import ExactLikelihood

    class CountingFitter(LegacyEMLBFGS):
        optimizer_count = 0

        def _new_optimizer(self, trainable):
            self.optimizer_count += 1
            return super()._new_optimizer(trainable)

    fitter = CountingFitter(
        LegacyEMLBFGSConfig(
            outer_iterations=2,
            e_steps_per_outer=1,
            decrease_e_steps=False,
            background_tolerance=0.0,
            parameter_tolerance=0.0,
            lbfgs_max_iter=1,
            retain_history_across_e_steps=True,
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
    assert fitter.optimizer_count == 1


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


def test_parameter_monitor_can_print_every_n_iterations(capsys):
    model, catalog, domain, state = make_problem()
    fitter = LegacyEMLBFGS(
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
            print_parameters=True,
            print_every=2,
        )
    )

    fitter.fit(
        model,
        ExactLikelihood(target_chunk_size=2, source_chunk_size=2),
        catalog,
        domain,
        state,
    )

    output = capsys.readouterr().out
    assert "EM 001" not in output
    assert "EM 002" in output
    assert "EM 003" not in output
