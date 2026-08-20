"""Minimal external ETAS3D model built by extending TorchSTPP's ETAS2D.

Run from the repository root with::

    python examples/custom_etas3d.py examples/custom_etas3d.toml

The example deliberately does not import the built-in ``ETAS3D`` class.  It
adds two parameters and a normalized conditional-Beta depth kernel, while the
catalog pipeline, likelihood, inference, checkpointing, and outputs remain
provided by TorchSTPP.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from pathlib import Path
import sys
from typing import Any

import torch

from torchstpp import run_etas
from torchstpp.integration import InfiniteSpaceIntegrator, PolygonSpatialIntegrator
from torchstpp.models import ModelParameter, ParameterConfig
from torchstpp.models.backgrounds import BackgroundModel
from torchstpp.models.etas import ETAS2D, ETAS2DParameters


@dataclass(frozen=True)
class UserETAS3DParameters(ETAS2DParameters):
    eta: ParameterConfig
    delta: ParameterConfig

    @classmethod
    def defaults(cls) -> "UserETAS3DParameters":
        base = ETAS2DParameters.defaults()
        return cls(
            **{field.name: getattr(base, field.name) for field in fields(base)},
            eta=ParameterConfig(10.0, lower_bound=0.0, transform="square"),
            delta=ParameterConfig(0.1, lower_bound=0.0, transform="square"),
        )

    @classmethod
    def from_model_section(
        cls, section: Mapping[str, Any]
    ) -> "UserETAS3DParameters":
        defaults = cls.defaults()
        supplied = section.get("parameters", {})
        if not isinstance(supplied, Mapping):
            raise TypeError("model.parameters must be a TOML table")
        values = {}
        for field in fields(defaults):
            settings = supplied.get(field.name, {})
            if not isinstance(settings, Mapping):
                raise TypeError(f"model.parameters.{field.name} must be a TOML table")
            values[field.name] = replace(getattr(defaults, field.name), **settings)
        return cls(**values)


class UserETAS3D(ETAS2D):
    """User-defined 3-D extension; no TorchSTPP core subclass is modified."""

    def __init__(
        self,
        parameters: UserETAS3DParameters,
        *,
        maximum_depth: float,
        depth_boundary_offset: float,
        background: BackgroundModel,
        spatial_integrator: Any,
    ) -> None:
        if maximum_depth <= 0.0:
            raise ValueError("maximum_depth must be positive")
        if depth_boundary_offset <= 0.0 or 2.0 * depth_boundary_offset >= maximum_depth:
            raise ValueError(
                "depth_boundary_offset must be smaller than maximum_depth / 2"
            )
        super().__init__(
            parameters,
            background=background,
            spatial_integrator=spatial_integrator,
        )
        self.maximum_depth = float(maximum_depth)
        self.depth_boundary_offset = float(depth_boundary_offset)
        self.eta_parameter = ModelParameter(
            parameters.eta, lower_bound=0.0, transform="square"
        )
        self.delta_parameter = ModelParameter(
            parameters.delta, lower_bound=0.0, transform="square"
        )

    @property
    def eta(self) -> torch.Tensor:
        return self.eta_parameter()

    @property
    def delta(self) -> torch.Tensor:
        return self.delta_parameter()

    def parameter_values(self) -> dict[str, torch.Tensor]:
        values = super().parameter_values()
        values.update(eta=self.eta, delta=self.delta)
        return values

    @staticmethod
    def _require_depth(data: Any, role: str) -> None:
        if not hasattr(data, "depth") or data.depth is None:
            raise TypeError(f"3-D {role} must provide depth")
        if data.depth.shape != data.time.shape:
            raise ValueError(f"{role}.depth must match {role}.time")

    def depth_concentration(self, source: Any) -> torch.Tensor:
        self._require_depth(source, "source")
        magnitude_excess = source.magnitude - source.completeness_magnitude
        return self.eta * torch.exp(-self.delta * magnitude_excess)

    def depth_kernel(self, target: Any, source: Any) -> torch.Tensor:
        self._require_depth(target, "target")
        self._require_depth(source, "source")
        maximum = target.depth.new_tensor(self.maximum_depth)
        offset = target.depth.new_tensor(self.depth_boundary_offset)
        target_fraction = target.depth[:, None] / maximum
        target_fraction = torch.where(
            target.depth[:, None] <= 0.0,
            offset / maximum,
            torch.where(
                target.depth[:, None] >= maximum,
                1.0 - offset / maximum,
                target_fraction,
            ),
        )
        source_fraction = source.depth[None, :] / maximum
        concentration = self.depth_concentration(source)[None, :]
        shape_a = concentration * source_fraction + 1.0
        shape_b = concentration * (1.0 - source_fraction) + 1.0
        log_density = (
            (shape_a - 1.0) * torch.log(target_fraction)
            + (shape_b - 1.0) * torch.log1p(-target_fraction)
            + torch.lgamma(shape_a + shape_b)
            - torch.lgamma(shape_a)
            - torch.lgamma(shape_b)
            - torch.log(maximum)
        )
        return torch.exp(log_density)

    def pairwise_rate(self, target: Any, source: Any) -> torch.Tensor:
        return super().pairwise_rate(target, source) * self.depth_kernel(target, source)

    def integrated_source(
        self, source: Any, domain: Any, space_mass: torch.Tensor | None = None
    ) -> torch.Tensor:
        self._require_depth(source, "source")
        # The conditional-Beta density is normalized over [0, maximum_depth].
        return super().integrated_source(source, domain, space_mass)


def build_user_etas3d(
    model_section: Mapping[str, Any],
    background: BackgroundModel,
    config: Any,
) -> UserETAS3D:
    """Model-factory hook used by ``run_etas``; ``config`` is available if needed."""

    del config
    integration = model_section.get("integration", {})
    integration_type = str(integration.get("type", "polygon")).lower()
    if integration_type == "infinite_space":
        integrator = InfiniteSpaceIntegrator()
    elif integration_type == "polygon":
        integrator = PolygonSpatialIntegrator(
            subdivisions=int(integration.get("subdivisions", 1001)),
            source_chunk_size=int(integration.get("source_chunk_size", 4096)),
        )
    else:
        raise ValueError(f"unsupported integration type: {integration_type!r}")

    return UserETAS3D(
        UserETAS3DParameters.from_model_section(model_section),
        maximum_depth=float(model_section["maximum_depth"]),
        depth_boundary_offset=float(model_section.get("depth_boundary_offset", 0.5)),
        background=background,
        spatial_integrator=integrator,
    )


def main(config_path: str | Path) -> None:
    result = run_etas(
        config_path,
        model_factories={"custom_etas3d": build_user_etas3d},
    )
    parameters = {
        name: float(value.detach().cpu())
        for name, value in result.run.model.parameter_values().items()
    }
    print(f"model={type(result.run.model).__name__}")
    print(f"nll={result.run.fit_result.negative_log_likelihood:.12g}")
    print(f"parameters={parameters}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python examples/custom_etas3d.py CONFIG.toml")
    main(sys.argv[1])
