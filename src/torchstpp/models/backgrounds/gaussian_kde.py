from typing import Any

import torch

from ...integration import PolygonSpatialIntegrator, SpatialIntegrator
from .state import KDEState


class GaussianKDEBackground:
    """Adaptive 2-D Gaussian KDE background matching the legacy ETAS code."""

    def __init__(
        self,
        *,
        target_chunk_size: int = 4096,
        source_chunk_size: int = 4096,
        spatial_integrator: SpatialIntegrator | None = None,
    ) -> None:
        if target_chunk_size < 1 or source_chunk_size < 1:
            raise ValueError("chunk sizes must be positive")
        self.target_chunk_size = target_chunk_size
        self.source_chunk_size = source_chunk_size
        self.spatial_integrator = spatial_integrator or PolygonSpatialIntegrator(
            subdivisions=1001, source_chunk_size=source_chunk_size
        )

    @staticmethod
    def spatial_density(
        distance_squared: torch.Tensor, source: Any
    ) -> torch.Tensor:
        bandwidth = source.bandwidth
        while bandwidth.ndim < distance_squared.ndim:
            bandwidth = bandwidth.unsqueeze(0)
        bandwidth_squared = bandwidth.square()
        return torch.exp(-distance_squared / (2.0 * bandwidth_squared)) / (
            2.0 * torch.pi * bandwidth_squared
        )

    @staticmethod
    def spatial_angular_integral(
        radius: torch.Tensor, source: Any
    ) -> torch.Tensor:
        bandwidth = source.bandwidth
        while bandwidth.ndim < radius.ndim:
            bandwidth = bandwidth.unsqueeze(-1)
        return (
            1.0
            - torch.exp(-radius.square() / (2.0 * bandwidth.square()))
        ) / (2.0 * torch.pi)

    def evaluate(self, target: Any, state: KDEState) -> torch.Tensor:
        """Evaluate ``sum_i weights_i K(x-x_i,h_i) / duration``."""
        state.validate()
        if not hasattr(target, "position"):
            raise TypeError("target must provide position")
        if target.position.ndim != 2 or target.position.shape[1] != 2:
            raise ValueError("target.position must have shape [N_target, 2]")
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
            value = target_position.new_zeros(target_end - target_begin)
            for source_begin in range(0, source_count, self.source_chunk_size):
                source_end = min(source_begin + self.source_chunk_size, source_count)
                source_position = state.position[source_begin:source_end]
                displacement = (
                    target_position[:, None, :] - source_position[None, :, :]
                )
                distance_squared = displacement.square().sum(dim=-1)
                source_view = _KDESourceView(state, source_begin, source_end)
                density = self.spatial_density(distance_squared, source_view)
                value = value + (density * source_view.weights[None, :]).sum(dim=-1)
            output_blocks.append(value / duration)
        return torch.cat(output_blocks)

    def integrate(self, state: KDEState, domain: Any) -> torch.Tensor:
        """Integrate the KDE basis over the observation space-time domain."""
        state.validate()
        if not hasattr(domain, "time_start") or not hasattr(domain, "time_end"):
            raise TypeError("domain must provide time_start and time_end")
        spatial_mass = self.spatial_integrator.integrate(self, state, domain)
        observation_duration = torch.as_tensor(
            domain.time_end - domain.time_start,
            dtype=state.position.dtype,
            device=state.position.device,
        )
        kde_duration = torch.as_tensor(
            state.duration, dtype=state.position.dtype, device=state.position.device
        )
        return observation_duration / kde_duration * torch.sum(
            state.weights * spatial_mass
        )


class _KDESourceView:
    def __init__(self, state: KDEState, begin: int, end: int):
        self.bandwidth = state.bandwidth[begin:end]
        self.weights = state.weights[begin:end]
