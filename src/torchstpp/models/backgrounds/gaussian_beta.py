from typing import Any

import torch

from ...integration import PolygonSpatialIntegrator, SpatialIntegrator
from ..components import conditional_beta_depth_density
from .gaussian_kde import GaussianKDEBackground, _KDESourceView
from .state import KDEState


class GaussianBetaKDEBackground(GaussianKDEBackground):
    """Fortran-compatible horizontal Gaussian and conditional-Beta depth KDE."""

    def __init__(
        self,
        *,
        maximum_depth: float,
        depth_concentration: float,
        boundary_offset: float | None = None,
        target_chunk_size: int = 4096,
        source_chunk_size: int = 4096,
        spatial_integrator: SpatialIntegrator | None = None,
    ) -> None:
        super().__init__(
            target_chunk_size=target_chunk_size,
            source_chunk_size=source_chunk_size,
            spatial_integrator=spatial_integrator
            or PolygonSpatialIntegrator(
                subdivisions=1001, source_chunk_size=source_chunk_size
            ),
        )
        if maximum_depth <= 0:
            raise ValueError("maximum_depth must be positive")
        if depth_concentration <= 0:
            raise ValueError("depth_concentration must be positive")
        if boundary_offset is not None and (
            boundary_offset <= 0 or 2.0 * boundary_offset >= maximum_depth
        ):
            raise ValueError(
                "boundary_offset must be None or a positive value smaller "
                "than maximum_depth / 2"
            )
        self.maximum_depth = maximum_depth
        self.depth_concentration = depth_concentration
        self.boundary_offset = boundary_offset

    def evaluate(self, target: Any, state: KDEState) -> torch.Tensor:
        """Evaluate sum(phi_j * K_xy * K_z) / duration."""
        state.validate()
        if state.depth is None:
            raise ValueError("3-D KDEState must provide source depth")
        if not hasattr(target, "position") or not hasattr(target, "depth"):
            raise TypeError("3-D KDE target must provide position and depth")
        if target.position.ndim != 2 or target.position.shape[1] != 2:
            raise ValueError("target.position must have shape [N_target, 2]")
        if target.depth.shape != (target.position.shape[0],):
            raise ValueError("target.depth must have shape [N_target]")

        duration = torch.as_tensor(
            state.duration,
            dtype=target.position.dtype,
            device=target.position.device,
        )
        output_blocks = []
        target_count = target.position.shape[0]
        source_count = state.position.shape[0]
        for target_begin in range(0, target_count, self.target_chunk_size):
            target_end = min(target_begin + self.target_chunk_size, target_count)
            target_position = target.position[target_begin:target_end]
            target_depth = target.depth[target_begin:target_end]
            value = target_position.new_zeros(target_end - target_begin)
            for source_begin in range(0, source_count, self.source_chunk_size):
                source_end = min(source_begin + self.source_chunk_size, source_count)
                displacement = (
                    target_position[:, None, :]
                    - state.position[None, source_begin:source_end, :]
                )
                distance_squared = displacement.square().sum(dim=-1)
                source_view = _KDESourceView(state, source_begin, source_end)
                horizontal_density = self.spatial_density(
                    distance_squared, source_view
                )
                concentration = target_position.new_tensor(
                    self.depth_concentration
                )
                depth_density = conditional_beta_depth_density(
                    target_depth[:, None],
                    state.depth[None, source_begin:source_end],
                    self.maximum_depth,
                    concentration,
                    boundary_offset=self.boundary_offset,
                )
                value = value + (
                    horizontal_density
                    * depth_density
                    * source_view.weights[None, :]
                ).sum(dim=-1)
            output_blocks.append(value / duration)
        return torch.cat(output_blocks)

    def integrate(self, state: KDEState, domain: Any) -> torch.Tensor:
        """Integrate over polygon x full depth interval [0, maximum_depth]."""
        if state.depth is None:
            raise ValueError("3-D KDEState must provide source depth")
        return super().integrate(state, domain)
