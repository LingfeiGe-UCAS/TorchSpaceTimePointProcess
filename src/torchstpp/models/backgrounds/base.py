from typing import Any, Protocol

import torch


class BackgroundModel(Protocol):
    def evaluate(self, target: Any, state: Any) -> torch.Tensor:
        """Return background basis values with shape [N_target]."""
        ...

    def integrate(self, state: Any, domain: Any) -> torch.Tensor:
        """Return the scalar background basis integral."""
        ...
