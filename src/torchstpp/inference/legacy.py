from dataclasses import dataclass
from typing import Any

import torch

from ..models.backgrounds import KDEState
from .em import update_background_probabilities
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
    lbfgs_history_size: int = 10
    line_search_fn: str | None = "strong_wolfe"
    target_chunk_size: int = 2048
    source_chunk_size: int = 4096
    print_parameters: bool = False
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
        }
        for name, value in integer_values.items():
            if value < 1:
                raise ValueError(f"{name} must be positive")
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
    def _background_values(model: Any, catalog: Any, state: KDEState) -> torch.Tensor:
        with torch.no_grad():
            return model.background.evaluate(catalog, state)

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
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not trainable:
            raise ValueError("model has no trainable parameters")

        broadcast_parameters = getattr(likelihood, "broadcast_parameters_", None)
        broadcast_background = getattr(
            likelihood, "broadcast_background_state_", None
        )
        synchronize_failure = getattr(likelihood, "synchronize_failure", None)
        validate_replicas = getattr(likelihood, "validate_replicas_", None)
        global_max = getattr(likelihood, "global_max", None)
        distributed_backward = getattr(
            likelihood, "backward_and_sum_gradients_", None
        )
        is_main_process = bool(
            getattr(likelihood, "is_main_process", True)
        )

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

        if broadcast_parameters is not None:
            broadcast_parameters(model)
        if broadcast_background is not None:
            broadcast_background(background_state, trainable[0])
        if validate_replicas is not None:
            validate_replicas(model, catalog, domain, background_state)

        state = background_state
        history: list[OptimizationStep] = []
        converged = False
        stop_reason: str | None = None

        for outer in range(self.config.outer_iterations):
            optimization_values = getattr(
                model, "optimization_parameter_values", model.parameter_values
            )
            previous_parameters = {
                name: value.detach().clone()
                for name, value in optimization_values().items()
            }
            previous_background = synchronized_call(
                lambda: self._background_values(model, catalog, state),
                "previous background evaluation",
            )
            assert previous_background is not None
            e_steps = self.config.e_steps_per_outer
            if self.config.decrease_e_steps:
                e_steps = max(e_steps - outer, self.config.minimum_e_steps)
            for _ in range(e_steps):
                optimization_state = state
                updated_state = synchronized_call(
                    lambda: update_background_probabilities(
                        model,
                        catalog,
                        state,
                        target_chunk_size=self.config.target_chunk_size,
                        source_chunk_size=self.config.source_chunk_size,
                    ),
                    "EM background-probability update",
                )
                assert updated_state is not None
                state = updated_state
                if broadcast_background is not None:
                    broadcast_background(state, trainable[0])

            # Fortran order is bkgdcalc(old phi) -> probcalc(new phi) -> dav().
            # Therefore dav() uses the KDE saved immediately before the final
            # probability update, while new phi is carried to the next outer
            # iteration. The ordinary implementation evaluates KDE from the
            # newly updated phi instead.
            if not fortran_em:
                optimization_state = state
            current_background = synchronized_call(
                lambda: self._background_values(model, catalog, state),
                "current background evaluation",
            )
            assert current_background is not None
            background_change = float(
                torch.max(torch.abs(current_background - previous_background)).item()
            )
            if global_max is not None:
                background_change = global_max(background_change, trainable[0])

            # The KDE update changes the objective between outer iterations.
            # L-BFGS curvature pairs are valid only for the objective that
            # produced them, so retain the model parameters but start each
            # fixed-background M step with empty curvature history. This also
            # matches Fortran davidn(), which initializes H = I on every call.
            optimizer = self._new_optimizer(trainable)

            def closure() -> torch.Tensor:
                optimizer.zero_grad()
                loss = likelihood(model, catalog, domain, optimization_state)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        "negative log-likelihood is not finite"
                    )
                if distributed_backward is None:
                    loss.backward()
                else:
                    distributed_backward(loss, model)
                return loss

            synchronized_call(
                lambda: optimizer.step(closure),
                "LBFGS parameter update",
            )
            if broadcast_parameters is not None:
                broadcast_parameters(model)
            with torch.no_grad():
                final_loss = synchronized_call(
                    lambda: likelihood(
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
                )
            )
            if self.config.print_parameters and is_main_process:
                formatted = " ".join(
                    f"{name}={value:.9g}"
                    for name, value in parameter_values.items()
                )
                print(
                    f"EM {outer + 1:03d} NLL={float(final_loss.item()):.12g} "
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
    """Fortran point-source EM ordering with PyTorch LBFGS optimization.

    The Fortran driver always executes every configured outer iteration. Its
    final ``bkgdcalc`` in an E-step sequence precedes ``probcalc``, so parameter
    optimization uses a KDE lagged by one probability update. ``rescale`` is a
    no-op for point-source catalogs and is intentionally not represented here.
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
