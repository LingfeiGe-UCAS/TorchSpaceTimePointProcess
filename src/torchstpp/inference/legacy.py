from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch

from ..finite_source.em import update_finite_source_weights
from ..models.backgrounds import KDEState
from .em import (
    BackgroundProbabilityUpdate,
    update_background_probabilities,
    update_background_probabilities_distributed,
)
from .result import FitResult, OptimizationStep


@dataclass(frozen=True)
class LegacyEMLBFGSConfig:
    """Settings for the legacy-compatible alternating ETAS optimizer."""

    outer_iterations: int = 11
    e_steps_per_outer: int = 10
    decrease_e_steps: bool = True
    minimum_e_steps: int = 1
    background_tolerance: float = 1e-5
    parameter_tolerance: float = 1e-5
    lbfgs_lr: float = 5e-2
    lbfgs_max_iter: int = 20
    lbfgs_max_iter_schedule: tuple[int, ...] | None = None
    lbfgs_max_eval: int | None = None
    lbfgs_max_eval_schedule: tuple[int, ...] | None = None
    lbfgs_history_size: int = 10
    retain_lbfgs_history_across_outer_iterations: bool = False
    # Deprecated compatibility alias retained for existing Python callers.
    retain_history_across_e_steps: bool | None = None
    line_search_fn: str | None = "strong_wolfe"
    target_chunk_size: int = 2048
    source_chunk_size: int = 4096
    print_parameters: bool = False
    print_every: int = 1
    stop_on_degenerate: bool = False
    boundary_tolerance: float = 1e-10
    maximum_absolute_value: float = 1e6

    def validate(self) -> None:
        integer_values = {
            "outer_iterations": self.outer_iterations,
            "e_steps_per_outer": self.e_steps_per_outer,
            "minimum_e_steps": self.minimum_e_steps,
            "lbfgs_max_iter": self.lbfgs_max_iter,
            "lbfgs_history_size": self.lbfgs_history_size,
            "target_chunk_size": self.target_chunk_size,
            "source_chunk_size": self.source_chunk_size,
            "print_every": self.print_every,
        }
        for name, value in integer_values.items():
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if self.lbfgs_max_iter_schedule is not None:
            if len(self.lbfgs_max_iter_schedule) < self.outer_iterations:
                raise ValueError(
                    "lbfgs_max_iter_schedule must provide at least one value "
                    "per outer iteration"
                )
            if any(value < 1 for value in self.lbfgs_max_iter_schedule):
                raise ValueError(
                    "all lbfgs_max_iter_schedule values must be positive"
                )
        if self.lbfgs_max_eval is not None and self.lbfgs_max_eval < 1:
            raise ValueError("lbfgs_max_eval must be positive")
        if self.lbfgs_max_eval_schedule is not None:
            if len(self.lbfgs_max_eval_schedule) < self.outer_iterations:
                raise ValueError(
                    "lbfgs_max_eval_schedule must provide at least one value "
                    "per outer iteration"
                )
            if any(value < 1 for value in self.lbfgs_max_eval_schedule):
                raise ValueError(
                    "all lbfgs_max_eval_schedule values must be positive"
                )
        if self.minimum_e_steps > self.e_steps_per_outer:
            raise ValueError("minimum_e_steps cannot exceed e_steps_per_outer")
        if self.background_tolerance < 0:
            raise ValueError("background_tolerance must be non-negative")
        if self.parameter_tolerance < 0:
            raise ValueError("parameter_tolerance must be non-negative")
        if self.lbfgs_lr <= 0:
            raise ValueError("lbfgs_lr must be positive")
        if self.boundary_tolerance < 0:
            raise ValueError("boundary_tolerance must be non-negative")
        if self.maximum_absolute_value <= 0:
            raise ValueError("maximum_absolute_value must be positive")


class LegacyEMLBFGS:
    """Alternate legacy phi/KDE updates with continuous-parameter LBFGS."""

    def __init__(self, config: LegacyEMLBFGSConfig | None = None) -> None:
        self.config = config or LegacyEMLBFGSConfig()
        self.config.validate()

    @staticmethod
    def _background_values(
        model: Any,
        catalog: Any,
        state: KDEState,
        target_ranges: Iterable[tuple[int, int]] | None = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            if target_ranges is None:
                return model.background.evaluate(catalog, state)
            blocks = [
                model.background.evaluate(catalog.take(slice(begin, end)), state)
                for begin, end in target_ranges
                if end > begin
            ]
            if not blocks:
                return catalog.time.new_empty((0,))
            return torch.cat(blocks)

    def _new_optimizer(
        self, trainable: list[torch.nn.Parameter]
    ) -> torch.optim.LBFGS:
        """Create fresh curvature history for the current fixed background."""

        return torch.optim.LBFGS(
            trainable,
            lr=self.config.lbfgs_lr,
            max_iter=self.config.lbfgs_max_iter,
            history_size=self.config.lbfgs_history_size,
            line_search_fn=self.config.line_search_fn,
        )

    def _degenerate_reason(
        self, model: Any, parameter_values: dict[str, float]
    ) -> str | None:
        """Describe an obviously degenerate physical parameter state."""

        for name, value in parameter_values.items():
            if not torch.isfinite(torch.tensor(value)):
                return f"parameter {name} is non-finite ({value})"
            if abs(value) >= self.config.maximum_absolute_value:
                return (
                    f"parameter {name} reached |value| >= "
                    f"{self.config.maximum_absolute_value:g} ({value:.9g})"
                )
            parameter_module = getattr(model, f"{name}_parameter", None)
            if parameter_module is None:
                continue
            lower = getattr(parameter_module, "lower_bound", None)
            upper = getattr(parameter_module, "upper_bound", None)
            tolerance = self.config.boundary_tolerance
            if lower is not None and value - float(lower) <= tolerance:
                return (
                    f"parameter {name} approached lower bound {lower:g} "
                    f"within {tolerance:g} ({value:.9g})"
                )
            if upper is not None and float(upper) - value <= tolerance:
                return (
                    f"parameter {name} approached upper bound {upper:g} "
                    f"within {tolerance:g} ({value:.9g})"
                )
        return None

    def fit(
        self,
        model: Any,
        likelihood: Any,
        catalog: Any,
        domain: Any,
        background_state: KDEState,
    ) -> FitResult:
        return self._fit(
            model,
            likelihood,
            catalog,
            domain,
            background_state,
            fortran_em=False,
            fixed_outer_iterations=False,
        )

    def _fit(
        self,
        model: Any,
        likelihood: Any,
        catalog: Any,
        domain: Any,
        background_state: KDEState,
        *,
        fortran_em: bool,
        fixed_outer_iterations: bool,
    ) -> FitResult:
        broadcast_parameters = getattr(likelihood, "broadcast_parameters_", None)
        broadcast_background = getattr(
            likelihood, "broadcast_background_state_", None
        )
        synchronize_failure = getattr(likelihood, "synchronize_failure", None)
        validate_replicas = getattr(likelihood, "validate_replicas_", None)
        global_max = getattr(likelihood, "global_max", None)
        local_target_ranges = getattr(likelihood, "local_target_ranges", None)
        local_target_schedule = getattr(likelihood, "local_target_schedule", None)
        global_sum = getattr(likelihood, "global_sum_", None)
        distributed_backward = getattr(
            likelihood, "backward_and_sum_gradients_", None
        )
        distributed_optimizer_step = getattr(likelihood, "run_lbfgs_step_", None)
        distributed_e_step = (
            (callable(local_target_schedule) or callable(local_target_ranges))
            and callable(global_sum)
            and callable(global_max)
        )
        is_main_process = bool(
            getattr(likelihood, "is_main_process", True)
        )

        # DistributedExactLikelihood validates names, shapes and requires_grad
        # flags before broadcasting. Do this before the local empty-list check
        # so mismatched ranks cannot diverge before the first collective.
        if broadcast_parameters is not None:
            broadcast_parameters(model)
        trainable = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        if not trainable:
            raise ValueError("model has no trainable parameters")

        def synchronized_call(operation, stage: str):
            value = None
            error: Exception | None = None
            try:
                value = operation()
            except Exception as caught:
                error = caught
            if synchronize_failure is None:
                if error is not None:
                    raise error
            else:
                synchronize_failure(error, trainable[0], stage=stage)
            return value

        if broadcast_background is not None:
            broadcast_background(background_state, trainable[0])
        if validate_replicas is not None:
            validate_replicas(model, catalog, domain, background_state)

        background_target_ranges = None
        e_step_target_chunk_size = self.config.target_chunk_size
        e_step_source_chunk_size = self.config.source_chunk_size
        if distributed_e_step:
            def build_e_step_schedule():
                if callable(local_target_schedule):
                    return local_target_schedule(
                        catalog.time,
                        catalog.time,
                        background_source_count=int(
                            background_state.position.shape[0]
                        ),
                    )
                return (
                    local_target_ranges(
                        catalog.time,
                        catalog.time,
                        background_source_count=int(
                            background_state.position.shape[0]
                        ),
                    ),
                    self.config.target_chunk_size,
                    self.config.source_chunk_size,
                )

            e_step_schedule = synchronized_call(
                build_e_step_schedule,
                "distributed E-step target scheduling",
            )
            assert e_step_schedule is not None
            (
                background_target_ranges,
                e_step_target_chunk_size,
                e_step_source_chunk_size,
            ) = e_step_schedule

        state = background_state
        history: list[OptimizationStep] = []
        converged = False
        stop_reason: str | None = None

        # Reusing a single optimizer deliberately carries L-BFGS curvature
        # pairs across changing fixed-background objectives. This is an
        # opt-in acceleration experiment; the mathematically conservative
        # default creates a fresh optimizer for every outer iteration.
        retain_history = (
            self.config.retain_lbfgs_history_across_outer_iterations
            or bool(self.config.retain_history_across_e_steps)
        )
        shared_optimizer = (
            self._new_optimizer(trainable)
            if retain_history
            else None
        )

        for outer in range(self.config.outer_iterations):
            current_max_iter = self.config.lbfgs_max_iter
            if self.config.lbfgs_max_iter_schedule is not None:
                current_max_iter = self.config.lbfgs_max_iter_schedule[outer]
            current_max_eval = (
                self.config.lbfgs_max_eval
                if self.config.lbfgs_max_eval is not None
                else current_max_iter * 5 // 4
            )
            if self.config.lbfgs_max_eval_schedule is not None:
                current_max_eval = self.config.lbfgs_max_eval_schedule[outer]
            optimization_values = getattr(
                model, "optimization_parameter_values", model.parameter_values
            )
            previous_parameters = {
                name: value.detach().clone()
                for name, value in optimization_values().items()
            }
            previous_background = synchronized_call(
                lambda: self._background_values(
                    model,
                    catalog,
                    state,
                    background_target_ranges,
                ),
                "previous background evaluation",
            )
            assert previous_background is not None
            e_steps = self.config.e_steps_per_outer
            if self.config.decrease_e_steps:
                e_steps = max(e_steps - outer, self.config.minimum_e_steps)
            for _ in range(e_steps):
                optimization_state = state
                # Transient inference result: the model interface is unchanged.
                reuse_intensity = callable(getattr(model, 'finite_source_parent_indices', None))
                if distributed_e_step:
                    updated_state = update_background_probabilities_distributed(
                        model,
                        catalog,
                        state,
                        likelihood,
                        target_chunk_size=e_step_target_chunk_size,
                        source_chunk_size=e_step_source_chunk_size,
                        target_ranges=background_target_ranges,
                        return_intensity=reuse_intensity,
                    )
                else:
                    updated_state = synchronized_call(
                        lambda: update_background_probabilities(
                            model,
                            catalog,
                            state,
                            target_chunk_size=self.config.target_chunk_size,
                            source_chunk_size=self.config.source_chunk_size,
                            return_intensity=reuse_intensity,
                        ),
                        "EM background-probability update",
                    )
                assert updated_state is not None
                event_intensity = None
                if isinstance(updated_state, BackgroundProbabilityUpdate):
                    event_intensity = updated_state.event_intensity
                    updated_state = updated_state.background_state
                if distributed_e_step:
                    update_finite_source_weights(
                        model,
                        catalog,
                        optimization_state,
                        target_chunk_size=e_step_target_chunk_size,
                        source_chunk_size=e_step_source_chunk_size,
                        target_ranges=background_target_ranges,
                        distributed_likelihood=likelihood,
                        event_intensity=event_intensity,
                    )
                else:
                    synchronized_call(
                        lambda: update_finite_source_weights(
                            model,
                            catalog,
                            optimization_state,
                            target_chunk_size=self.config.target_chunk_size,
                            source_chunk_size=self.config.source_chunk_size,
                            event_intensity=event_intensity,
                        ),
                        "finite-source E-step and rescale",
                    )
                state = updated_state
                if broadcast_background is not None and not distributed_e_step:
                    broadcast_background(state, trainable[0])

            # Fortran order is bkgdcalc(old phi) -> probcalc(new phi) -> dav().
            # Therefore dav() uses the KDE saved immediately before the final
            # probability update, while new phi is carried to the next outer
            # iteration. The ordinary implementation evaluates KDE from the
            # newly updated phi instead.
            if not fortran_em:
                optimization_state = state
            current_background = synchronized_call(
                lambda: self._background_values(
                    model,
                    catalog,
                    state,
                    background_target_ranges,
                ),
                "current background evaluation",
            )
            assert current_background is not None
            background_change = (
                0.0
                if current_background.numel() == 0
                else float(
                    torch.max(
                        torch.abs(current_background - previous_background)
                    ).item()
                )
            )
            if global_max is not None:
                background_change = global_max(background_change, trainable[0])

            # By default the KDE update invalidates the previous objective's
            # curvature pairs, so each M step starts with empty history. The
            # explicit retention mode instead reuses the optimizer and its
            # bounded L-BFGS history as an empirical acceleration strategy.
            optimizer = shared_optimizer or self._new_optimizer(trainable)
            optimizer.param_groups[0]["max_iter"] = current_max_iter
            # Match torch.optim.LBFGS's default relation when max_iter varies
            # by outer iteration instead of retaining the constructor default.
            optimizer.param_groups[0]["max_eval"] = current_max_eval
            optimizer_state = optimizer.state.get(trainable[0], {})
            previous_m_step_iterations = int(optimizer_state.get("n_iter", 0))
            previous_m_step_evaluations = int(
                optimizer_state.get("func_evals", 0)
            )

            prepare_m_step = getattr(likelihood, "prepare_m_step", None)
            m_step_likelihood = likelihood
            if callable(prepare_m_step):
                m_step_likelihood = synchronized_call(
                    lambda: prepare_m_step(model, catalog, domain, optimization_state),
                    "fixed M-step background preparation",
                )

            def closure() -> torch.Tensor:
                synchronized_call(optimizer.zero_grad, "LBFGS gradient reset")
                loss = m_step_likelihood(model, catalog, domain, optimization_state)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        "negative log-likelihood is not finite"
                    )
                if distributed_backward is None:
                    loss.backward()
                else:
                    distributed_backward(loss, model)
                return loss

            if callable(distributed_optimizer_step):
                counts = synchronized_call(
                    lambda: distributed_optimizer_step(optimizer, model, closure),
                    "coordinated LBFGS parameter update",
                )
                assert counts is not None
                m_step_iterations, m_step_function_evaluations = counts
            else:
                synchronized_call(
                    lambda: optimizer.step(closure),
                    "LBFGS parameter update",
                )
                optimizer_state = optimizer.state[trainable[0]]
                m_step_iterations = (
                    int(optimizer_state.get("n_iter", 0))
                    - previous_m_step_iterations
                )
                m_step_function_evaluations = (
                    int(optimizer_state.get("func_evals", 0))
                    - previous_m_step_evaluations
                )
            if broadcast_parameters is not None:
                broadcast_parameters(model)
            with torch.no_grad():
                final_loss = synchronized_call(
                    lambda: m_step_likelihood(
                        model, catalog, domain, optimization_state
                    ),
                    "final likelihood evaluation",
                )
                assert final_loss is not None
                if not torch.isfinite(final_loss):
                    raise FloatingPointError(
                        "LBFGS accepted a non-finite final objective"
                    )
            parameter_values = {
                name: float(value.detach().item())
                for name, value in model.parameter_values().items()
            }
            current_optimization_values = optimization_values()
            parameter_change = max(
                float(
                    (
                        torch.abs(current_optimization_values[name].detach() - previous)
                        / (1.0 + torch.abs(previous))
                    ).item()
                )
                for name, previous in previous_parameters.items()
            )
            if global_max is not None:
                parameter_change = global_max(parameter_change, trainable[0])
            history.append(
                OptimizationStep(
                    iteration=outer + 1,
                    negative_log_likelihood=float(final_loss.item()),
                    background_change=background_change,
                    parameter_change=parameter_change,
                    parameter_values=parameter_values,
                    m_step_max_iter=current_max_iter,
                    m_step_max_eval=current_max_eval,
                    m_step_iterations=m_step_iterations,
                    m_step_function_evaluations=m_step_function_evaluations,
                )
            )
            if (
                self.config.print_parameters
                and is_main_process
                and (outer + 1) % self.config.print_every == 0
            ):
                formatted = " ".join(
                    f"{name}={value:.9g}"
                    for name, value in parameter_values.items()
                )
                print(
                    f"EM {outer + 1:03d} NLL={float(final_loss.item()):.12g} "
                    f"M_budget={current_max_iter} "
                    f"M_eval_budget={current_max_eval} "
                    f"M_iterations={m_step_iterations} "
                    f"M_evaluations={m_step_function_evaluations} "
                    f"background_change={background_change:.6g} "
                    f"parameter_change={parameter_change:.6g} {formatted}",
                    flush=True,
                )
            if self.config.stop_on_degenerate:
                stop_reason = self._degenerate_reason(model, parameter_values)
                if stop_reason is not None:
                    if is_main_process:
                        print(
                            f"Stopping after EM {outer + 1}: "
                            f"degenerate solution: {stop_reason}",
                            flush=True,
                        )
                    break
            if (
                background_change <= self.config.background_tolerance
                and parameter_change <= self.config.parameter_tolerance
            ):
                converged = True
                if not fixed_outer_iterations:
                    break

        return FitResult(
            negative_log_likelihood=history[-1].negative_log_likelihood,
            background_state=state,
            history=tuple(history),
            converged=converged,
            outer_iterations=len(history),
            stop_reason=stop_reason,
        )


class FortranEMLBFGS(LegacyEMLBFGS):
    """Fortran EM ordering with PyTorch LBFGS optimization.

    The Fortran driver always executes every configured outer iteration. Its
    final ``bkgdcalc`` in an E-step sequence precedes ``probcalc``, so parameter
    optimization uses a KDE lagged by one probability update.  Finite-source
    models additionally run the legacy subfault ``rescale`` after every
    probability update; it remains an exact no-op for point-source models.
    """

    def fit(
        self,
        model: Any,
        likelihood: Any,
        catalog: Any,
        domain: Any,
        background_state: KDEState,
    ) -> FitResult:
        return self._fit(
            model,
            likelihood,
            catalog,
            domain,
            background_state,
            fortran_em=True,
            fixed_outer_iterations=True,
        )
