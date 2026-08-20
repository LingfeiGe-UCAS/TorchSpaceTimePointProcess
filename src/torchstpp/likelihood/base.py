from typing import Any, Protocol

import torch


class Likelihood(Protocol):
    def __call__(
        self,
        model: Any,
        catalog: Any,
        domain: Any,
        background_state: Any,
    ) -> torch.Tensor:
        """Return a scalar negative log-likelihood."""
        ...
