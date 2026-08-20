from typing import Any, Protocol

import torch


class SpatialIntegrator(Protocol):
    def integrate(self, model: Any, source: Any, domain: Any) -> torch.Tensor:
        """Return one spatial mass value for each source event."""
        ...
