from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ParameterConfig:
    """Initial value, trainability, and optional bounds for one parameter."""

    value: float
    trainable: bool = True
    lower_bound: float | None = None
    upper_bound: float | None = None
    transform: str | None = None


class ModelParameter(nn.Module):
    """Store an unconstrained raw parameter and expose its constrained value."""

    def __init__(
        self,
        config: ParameterConfig,
        *,
        lower_bound: float | None = None,
        upper_bound: float | None = None,
        transform: str = "softplus",
    ) -> None:
        super().__init__()

        lower = config.lower_bound if lower_bound is None else lower_bound
        upper = config.upper_bound if upper_bound is None else upper_bound
        if lower is not None and upper is not None and lower >= upper:
            raise ValueError("lower_bound must be smaller than upper_bound")

        resolved_transform = config.transform or transform
        if resolved_transform not in {"softplus", "square", "centered_log"}:
            raise ValueError(
                "parameter transform must be 'softplus', 'square', or "
                "'centered_log'"
            )
        if resolved_transform == "square" and (lower is None) == (upper is None):
            raise ValueError(
                "the square transform requires exactly one bound"
            )
        if resolved_transform == "centered_log" and not (
            lower is not None and upper is None
        ):
            raise ValueError(
                "the centered_log transform requires a lower bound and no "
                "upper bound"
            )

        value = torch.tensor(config.value, dtype=torch.get_default_dtype())
        self.lower_bound = lower
        self.upper_bound = upper
        self.transform = resolved_transform

        if lower is not None and value <= lower:
            raise ValueError(f"parameter value must be greater than {lower}")
        if upper is not None and value >= upper:
            raise ValueError(f"parameter value must be smaller than {upper}")

        self.reference_scale: float | None = None
        if resolved_transform == "square":
            distance = value - lower if lower is not None else upper - value
            raw_value = torch.sqrt(distance)
        elif resolved_transform == "centered_log":
            # Automatically center the dimensionless optimization coordinate
            # at zero without changing the configured physical initial value.
            self.reference_scale = float((value - lower).item())
            raw_value = torch.zeros_like(value)
        elif lower is None and upper is None:
            raw_value = value
        elif upper is None:
            shifted = value - lower
            raw_value = shifted + torch.log(-torch.expm1(-shifted))
        elif lower is None:
            shifted = upper - value
            raw_value = -(shifted + torch.log(-torch.expm1(-shifted)))
        else:
            ratio = (value - lower) / (upper - lower)
            raw_value = torch.logit(ratio)

        self.raw = nn.Parameter(raw_value, requires_grad=config.trainable)

    def forward(self) -> torch.Tensor:
        if self.transform == "square":
            if self.lower_bound is not None:
                return self.raw.new_tensor(self.lower_bound) + self.raw.square()
            return self.raw.new_tensor(self.upper_bound) - self.raw.square()
        if self.transform == "centered_log":
            return (
                self.raw.new_tensor(self.lower_bound)
                + self.raw.new_tensor(self.reference_scale) * torch.exp(self.raw)
            )
        if self.lower_bound is None and self.upper_bound is None:
            return self.raw
        if self.upper_bound is None:
            return self.raw.new_tensor(self.lower_bound) + F.softplus(self.raw)
        if self.lower_bound is None:
            return self.raw.new_tensor(self.upper_bound) - F.softplus(-self.raw)

        lower = self.raw.new_tensor(self.lower_bound)
        upper = self.raw.new_tensor(self.upper_bound)
        return lower + (upper - lower) * torch.sigmoid(self.raw)
