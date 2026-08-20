from collections.abc import Callable, Mapping
from typing import Any

import torch
from torch import nn

from .parameters import ModelParameter, ParameterConfig

ModelFunction = Callable[..., torch.Tensor]


class CustomPointProcessModel(nn.Module):
    """A point-process model assembled from parameters and user functions."""

    def __init__(
        self,
        *,
        parameters: Mapping[str, ParameterConfig],
        background: ModelFunction,
        trigger_kernel: ModelFunction,
        background_integral: ModelFunction,
        source_integral: ModelFunction,
    ) -> None:
        super().__init__()
        if not parameters:
            raise ValueError("parameters cannot be empty")

        self.parameter_store = nn.ModuleDict(
            {name: ModelParameter(config) for name, config in parameters.items()}
        )
        self._background = background
        self._trigger_kernel = trigger_kernel
        self._background_integral = background_integral
        self._source_integral = source_integral

    def parameter_values(self) -> dict[str, torch.Tensor]:
        return {name: parameter() for name, parameter in self.parameter_store.items()}

    def background_rate(self, target: Any) -> torch.Tensor:
        """Return the background rate for each target event."""
        return self._background(target, self.parameter_values())

    def pairwise_rate(self, target: Any, source: Any) -> torch.Tensor:
        """Return the triggering rate for the requested target-source pairs."""
        return self._trigger_kernel(target, source, self.parameter_values())

    def integrated_background(self, domain: Any) -> torch.Tensor:
        """Return the scalar background compensator over the domain."""
        return self._background_integral(domain, self.parameter_values())

    def integrated_source(self, source: Any, domain: Any) -> torch.Tensor:
        """Return one triggering compensator value for each source event."""
        return self._source_integral(source, domain, self.parameter_values())
