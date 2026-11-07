from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import nn

from ...integration import PolygonSpatialIntegrator, SpatialIntegrator
from ..backgrounds import BackgroundModel, GaussianKDEBackground
from ..parameters import ModelParameter, ParameterConfig


@dataclass(frozen=True)
class ETAS2DParameters:
    mu: ParameterConfig
    amplitude: ParameterConfig
    alpha: ParameterConfig
    c: ParameterConfig
    p: ParameterConfig
    d: ParameterConfig
    q: ParameterConfig
    gamma: ParameterConfig

    @classmethod
    def defaults(cls) -> "ETAS2DParameters":
        return cls(
            mu=ParameterConfig(0.02, lower_bound=0.0),
            amplitude=ParameterConfig(0.15, lower_bound=0.0),
            alpha=ParameterConfig(1.0, lower_bound=0.0),
            c=ParameterConfig(0.01, lower_bound=0.0),
            p=ParameterConfig(1.15, lower_bound=1.0),
            d=ParameterConfig(0.5, lower_bound=0.0),
            q=ParameterConfig(1.5, lower_bound=1.0),
            gamma=ParameterConfig(0.8, lower_bound=0.0),
        )


class ETAS2D(nn.Module):
    """Standard normalized 2-D ETAS model containing only model state."""

    def __init__(
        self,
        parameters: ETAS2DParameters | None = None,
        *,
        background: BackgroundModel | None = None,
        spatial_integrator: SpatialIntegrator | None = None,
        minimum_spatial_scale: float = 0.0,
    ) -> None:
        super().__init__()
        parameters = parameters or ETAS2DParameters.defaults()
        minimum_spatial_scale = float(minimum_spatial_scale)
        if not math.isfinite(minimum_spatial_scale) or minimum_spatial_scale < 0.0:
            raise ValueError("minimum_spatial_scale must be finite and non-negative")
        self.background = (
            GaussianKDEBackground()
            if background is None
            else background
        )
        self.spatial_integrator = (
            PolygonSpatialIntegrator(subdivisions=1001)
            if spatial_integrator is None
            else spatial_integrator
        )
        self.mu_parameter = self._build_parameter(parameters.mu, 0.0)
        self.amplitude_parameter = self._build_parameter(parameters.amplitude, 0.0)
        self.alpha_parameter = self._build_parameter(parameters.alpha, 0.0)
        self.c_parameter = self._build_parameter(parameters.c, 0.0)
        self.p_parameter = self._build_parameter(parameters.p, 1.0)
        self.d_parameter = self._build_parameter(parameters.d, 0.0)
        self.q_parameter = self._build_parameter(parameters.q, 1.0)
        self.gamma_parameter = self._build_parameter(parameters.gamma, 0.0)
        self.register_buffer(
            "_minimum_spatial_scale",
            torch.tensor(minimum_spatial_scale),
        )

    @staticmethod
    def _build_parameter(
        config: ParameterConfig, default_lower_bound: float
    ) -> ModelParameter:
        lower_bound = (
            default_lower_bound
            if config.lower_bound is None
            else config.lower_bound
        )
        return ModelParameter(config, lower_bound=lower_bound)

    @staticmethod
    def _validate_source(source: Any) -> None:
        required = ("time", "position", "magnitude", "completeness_magnitude")
        missing = [name for name in required if not hasattr(source, name)]
        if missing:
            raise TypeError(f"source is missing required attributes: {missing}")

    @staticmethod
    def _validate_target(target: Any) -> None:
        required = ("time", "position")
        missing = [name for name in required if not hasattr(target, name)]
        if missing:
            raise TypeError(f"target is missing required attributes: {missing}")

    @staticmethod
    def _validate_domain(domain: Any) -> None:
        required = ("time_start", "time_end", "area")
        missing = [name for name in required if not hasattr(domain, name)]
        if missing:
            raise TypeError(f"domain is missing required attributes: {missing}")

    @property
    def mu(self): return self.mu_parameter()
    @property
    def amplitude(self): return self.amplitude_parameter()
    @property
    def alpha(self): return self.alpha_parameter()
    @property
    def c(self): return self.c_parameter()
    @property
    def p(self): return self.p_parameter()
    @property
    def d(self): return self.d_parameter()
    @property
    def q(self): return self.q_parameter()
    @property
    def gamma(self): return self.gamma_parameter()

    @property
    def minimum_spatial_scale(self) -> torch.Tensor:
        """Fixed squared-coordinate floor added to every spatial kernel scale."""

        return self._minimum_spatial_scale

    def parameter_values(self) -> dict[str, torch.Tensor]:
        return {name: getattr(self, name) for name in (
            "mu", "amplitude", "alpha", "c", "p", "d", "q", "gamma"
        )}

    def magnitude_productivity(self, source: Any) -> torch.Tensor:
        dm = source.magnitude - source.completeness_magnitude
        return self.amplitude * torch.exp(self.alpha * dm)

    def temporal_kernel(
        self,
        delta_time: torch.Tensor,
        causal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        p, c = self.p, self.c
        safe_delta = delta_time.clamp_min(0.0)
        rate = (p - 1.0) * torch.pow(1.0 + safe_delta / c, -p) / c
        if causal_mask is None:
            causal_mask = delta_time > 0.0
        elif causal_mask.shape != delta_time.shape or causal_mask.dtype != torch.bool:
            raise ValueError("causal_mask must be a bool tensor matching delta_time")
        return torch.where(causal_mask, rate, torch.zeros_like(rate))

    def temporal_integral(self, source_time, time_start, time_end) -> torch.Tensor:
        p, c = self.p, self.c
        start = torch.as_tensor(time_start, dtype=source_time.dtype, device=source_time.device)
        end = torch.as_tensor(time_end, dtype=source_time.dtype, device=source_time.device)
        lower = (start - source_time).clamp_min(0.0)
        upper = (end - source_time).clamp_min(0.0)
        value = torch.pow(1.0 + lower / c, 1.0 - p) - torch.pow(
            1.0 + upper / c, 1.0 - p
        )
        return torch.where(end > source_time, value, torch.zeros_like(value))

    def spatial_scale(self, source: Any) -> torch.Tensor:
        dm = source.magnitude - source.completeness_magnitude
        return self.minimum_spatial_scale + self.d * torch.exp(self.gamma * dm)

    @staticmethod
    def _match_dimensions(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        while value.ndim < reference.ndim:
            value = value.unsqueeze(-1)
        return value

    def spatial_kernel(self, distance_squared: torch.Tensor, source: Any) -> torch.Tensor:
        q = self.q
        # Pairwise matrices use [N_target, N_source], so the source dimension
        # is the last axis and sigma must broadcast as [1, N_source].
        scale = self.spatial_scale(source)
        while scale.ndim < distance_squared.ndim:
            scale = scale.unsqueeze(0)
        return (q - 1.0) * torch.pow(
            1.0 + distance_squared / scale, -q
        ) / (torch.pi * scale)

    def spatial_radial_integral(self, radius: torch.Tensor, source: Any) -> torch.Tensor:
        q = self.q
        scale = self._match_dimensions(self.spatial_scale(source), radius)
        return 1.0 - torch.pow(1.0 + radius.square() / scale, 1.0 - q)

    def spatial_angular_integral(self, radius: torch.Tensor, source: Any) -> torch.Tensor:
        """Radial primitive per radian; equivalent to old ``int_fr``."""
        return self.spatial_radial_integral(radius, source) / (2.0 * torch.pi)

    def background_rate(self, background_values: torch.Tensor) -> torch.Tensor:
        """Scale externally computed background values, as ``mu * bk_mu``."""
        return self.mu * background_values

    def pairwise_rate(self, target: Any, source: Any) -> torch.Tensor:
        """Return triggering contributions with shape [N_target, N_source]."""
        self._validate_target(target)
        self._validate_source(source)
        if target.position.ndim != 2 or target.position.shape[1] != 2:
            raise ValueError("target.position must have shape [N_target, 2]")
        if source.position.ndim != 2 or source.position.shape[1] != 2:
            raise ValueError("source.position must have shape [N_source, 2]")

        delta_time = target.time[:, None] - source.time[None, :]
        displacement = target.position[:, None, :] - source.position[None, :, :]
        distance_squared = displacement.square().sum(dim=-1)

        productivity = self.magnitude_productivity(source)[None, :]
        target_order = getattr(target, "sequence_index", None)
        source_order = getattr(source, "sequence_index", None)
        causal_mask = None
        if target_order is not None or source_order is not None:
            if target_order is None or source_order is None:
                raise ValueError(
                    "target and source must both provide sequence_index or neither"
                )
            causal_mask = source_order[None, :] < target_order[:, None]
        temporal = self.temporal_kernel(delta_time, causal_mask)
        spatial = self.spatial_kernel(distance_squared, source)
        return productivity * temporal * spatial

    def integrated_background(self, background_mass: torch.Tensor) -> torch.Tensor:
        """Scale the externally computed background integral by ``mu``."""
        if background_mass.ndim != 0:
            raise ValueError("background_mass must be a scalar tensor")
        return self.mu * background_mass

    def integrated_source(
        self,
        source: Any,
        domain: Any,
        space_mass: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._validate_source(source)
        self._validate_domain(domain)
        if space_mass is None:
            space_mass = self.spatial_integrator.integrate(self, source, domain)
        if space_mass.shape != source.time.shape:
            raise ValueError("space_mass must have shape [N_source]")
        time_mass = self.temporal_integral(
            source.time, domain.time_start, domain.time_end
        )
        return self.magnitude_productivity(source) * time_mass * space_mass

    def forward(
        self,
        target: Any,
        background_values: torch.Tensor,
        source: Any | None = None,
    ) -> torch.Tensor:
        background = self.background_rate(background_values)
        if source is None:
            return background
        return background + self.pairwise_rate(target, source).sum(dim=-1)
