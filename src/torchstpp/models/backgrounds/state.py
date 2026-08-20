from dataclasses import dataclass

import torch


@dataclass
class KDEState:
    """State used by the adaptive Gaussian KDE background."""

    position: torch.Tensor
    bandwidth: torch.Tensor
    weights: torch.Tensor
    duration: float | torch.Tensor
    depth: torch.Tensor | None = None

    def validate(self) -> None:
        if self.position.ndim != 2 or self.position.shape[1] != 2:
            raise ValueError("position must have shape [N_source, 2]")
        source_count = self.position.shape[0]
        if self.bandwidth.shape != (source_count,):
            raise ValueError("bandwidth must have shape [N_source]")
        if self.weights.shape != (source_count,):
            raise ValueError("weights must have shape [N_source]")
        if torch.any(self.bandwidth <= 0):
            raise ValueError("bandwidth must be positive")
        if self.depth is not None:
            if self.depth.shape != (source_count,):
                raise ValueError("depth must be None or have shape [N_source]")
            if self.depth.device != self.position.device:
                raise ValueError("depth and position must be on the same device")
        duration = torch.as_tensor(
            self.duration, dtype=self.position.dtype, device=self.position.device
        )
        if duration.ndim != 0 or duration <= 0:
            raise ValueError("duration must be a positive scalar")

    def to(self, *args, **kwargs) -> "KDEState":
        return KDEState(
            position=self.position.to(*args, **kwargs),
            bandwidth=self.bandwidth.to(*args, **kwargs),
            weights=self.weights.to(*args, **kwargs),
            duration=torch.as_tensor(self.duration).to(*args, **kwargs),
            depth=None if self.depth is None else self.depth.to(*args, **kwargs),
        )
