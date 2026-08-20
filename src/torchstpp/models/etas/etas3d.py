from dataclasses import dataclass
import math
from typing import Any

import torch

from ...integration import PolygonSpatialIntegrator, SpatialIntegrator
from ..backgrounds import BackgroundModel, GaussianBetaKDEBackground
from ..components import conditional_beta_depth_density
from ..parameters import ModelParameter, ParameterConfig
from .etas2d import ETAS2D, ETAS2DParameters


@dataclass(frozen=True)
class ETAS3DParameters(ETAS2DParameters):
    """Point-source 3-D ETAS parameters, excluding all finite-source terms."""

    eta: ParameterConfig
    delta: ParameterConfig

    @classmethod
    def defaults(cls) -> "ETAS3DParameters":
        base = ETAS2DParameters.defaults()
        return cls(
            mu=base.mu,
            amplitude=base.amplitude,
            alpha=base.alpha,
            c=base.c,
            p=base.p,
            d=base.d,
            q=base.q,
            gamma=base.gamma,
            eta=ParameterConfig(10.0, lower_bound=0.0),
            delta=ParameterConfig(0.1, lower_bound=0.0),
        )


class ETAS3D(ETAS2D):
    """Fortran-compatible point-source ETAS with a conditional-Beta depth pdf."""

    def __init__(
        self,
        parameters: ETAS3DParameters | None = None,
        *,
        maximum_depth: float,
        background_depth_concentration: float | None = None,
        depth_boundary_offset: float = 0.5,
        depth_reference_magnitude_offset: float = 0.0,
        background: BackgroundModel | None = None,
        spatial_integrator: SpatialIntegrator | None = None,
    ) -> None:
        parameters = parameters or ETAS3DParameters.defaults()
        if maximum_depth <= 0:
            raise ValueError("maximum_depth must be positive")
        if depth_boundary_offset <= 0 or 2.0 * depth_boundary_offset >= maximum_depth:
            raise ValueError(
                "depth_boundary_offset must be positive and smaller than maximum_depth / 2"
            )
        if not math.isfinite(depth_reference_magnitude_offset):
            raise ValueError("depth_reference_magnitude_offset must be finite")
        if depth_reference_magnitude_offset < 0.0:
            raise ValueError(
                "depth_reference_magnitude_offset must be non-negative"
            )
        spatial_integrator = spatial_integrator or PolygonSpatialIntegrator(
            subdivisions=1001
        )
        if background is None:
            if background_depth_concentration is None:
                raise ValueError(
                    "background_depth_concentration is required for the default "
                    "GaussianBetaKDEBackground"
                )
            background = GaussianBetaKDEBackground(
                maximum_depth=maximum_depth,
                depth_concentration=background_depth_concentration,
                boundary_offset=depth_boundary_offset,
            )
        super().__init__(
            parameters,
            background=background,
            spatial_integrator=spatial_integrator,
        )
        self.maximum_depth = maximum_depth
        self.depth_boundary_offset = depth_boundary_offset
        self.register_buffer(
            "_depth_reference_magnitude_offset",
            torch.tensor(float(depth_reference_magnitude_offset)),
        )
        eta_config = parameters.eta
        if depth_reference_magnitude_offset:
            if eta_config.upper_bound is not None or eta_config.lower_bound not in {
                None, 0.0
            }:
                raise ValueError(
                    "reference-magnitude depth parameterization supports eta "
                    "only with a zero lower bound and no upper bound"
                )
            reference_value = eta_config.value * math.exp(
                -parameters.delta.value * depth_reference_magnitude_offset
            )
            eta_config = ParameterConfig(
                value=reference_value,
                trainable=eta_config.trainable,
                lower_bound=0.0,
                transform=eta_config.transform,
            )
        self.eta_parameter = ModelParameter(
            eta_config, lower_bound=0.0
        )
        self.delta_parameter = ModelParameter(
            parameters.delta, lower_bound=0.0
        )

    @staticmethod
    def _validate_depth(data: Any, role: str) -> None:
        if not hasattr(data, "depth") or data.depth is None:
            raise TypeError(f"3-D {role} must provide depth")
        if data.depth.shape != data.time.shape:
            raise ValueError(f"{role}.depth must have shape [N_{role}]")

    @property
    def depth_reference_magnitude_offset(self) -> float:
        """Magnitude offset stored in model state for checkpoint-safe reloads."""
        return float(self._depth_reference_magnitude_offset.item())

    @property
    def eta(self) -> torch.Tensor:
        return self.depth_reference_concentration * torch.exp(
            self.delta * self.depth_reference_magnitude_offset
        )

    @property
    def depth_reference_concentration(self) -> torch.Tensor:
        """Depth concentration at ``Mc + depth_reference_magnitude_offset``."""
        return self.eta_parameter()

    @property
    def delta(self) -> torch.Tensor:
        return self.delta_parameter()

    def parameter_values(self) -> dict[str, torch.Tensor]:
        values = super().parameter_values()
        values.update(eta=self.eta, delta=self.delta)
        return values

    def optimization_parameter_values(self) -> dict[str, torch.Tensor]:
        """Return the coordinates that the optimizer actually updates."""
        values = super().parameter_values()
        if self.depth_reference_magnitude_offset:
            values.update(
                depth_reference_concentration=self.depth_reference_concentration,
                delta=self.delta,
            )
        else:
            values.update(eta=self.eta, delta=self.delta)
        return values

    def recenter_depth_reference_magnitude_offset(
        self, new_offset: float
    ) -> None:
        """Change the depth-kernel coordinate origin without changing the model."""
        new_offset = float(new_offset)
        if not math.isfinite(new_offset) or new_offset < 0.0:
            raise ValueError("new depth reference offset must be finite and non-negative")
        old_offset = self.depth_reference_magnitude_offset
        if new_offset == old_offset:
            return
        with torch.no_grad():
            new_reference = self.depth_reference_concentration * torch.exp(
                -self.delta * (new_offset - old_offset)
            )
        old_parameter = self.eta_parameter
        replacement = ModelParameter(
            ParameterConfig(
                value=float(new_reference.detach().cpu().item()),
                trainable=old_parameter.raw.requires_grad,
                lower_bound=0.0,
                transform=old_parameter.transform,
            ),
            lower_bound=0.0,
        ).to(device=old_parameter.raw.device, dtype=old_parameter.raw.dtype)
        self.eta_parameter = replacement
        self._depth_reference_magnitude_offset.fill_(new_offset)

    def depth_concentration(self, source: Any) -> torch.Tensor:
        """H_j = eta * exp(-delta * (M_j - Mc)), as in the Fortran PS branch."""
        self._validate_source(source)
        self._validate_depth(source, "source")
        magnitude_excess = source.magnitude - source.completeness_magnitude
        centered_magnitude = (
            magnitude_excess - self.depth_reference_magnitude_offset
        )
        return self.depth_reference_concentration * torch.exp(
            -self.delta * centered_magnitude
        )

    def depth_kernel(self, target: Any, source: Any) -> torch.Tensor:
        self._validate_target(target)
        self._validate_source(source)
        self._validate_depth(target, "target")
        self._validate_depth(source, "source")
        return conditional_beta_depth_density(
            target.depth[:, None],
            source.depth[None, :],
            self.maximum_depth,
            self.depth_concentration(source)[None, :],
            boundary_offset=self.depth_boundary_offset,
        )

    def pairwise_rate(self, target: Any, source: Any) -> torch.Tensor:
        horizontal_time_magnitude = super().pairwise_rate(target, source)
        return horizontal_time_magnitude * self.depth_kernel(target, source)

    def integrated_source(
        self,
        source: Any,
        domain: Any,
        space_mass: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Integrate over horizontal domain and the full depth range [0, Z]."""
        self._validate_depth(source, "source")
        # The conditional Beta depth density integrates to exactly one over
        # [0, maximum_depth], matching the Fortran point-source compensator.
        return super().integrated_source(source, domain, space_mass)
