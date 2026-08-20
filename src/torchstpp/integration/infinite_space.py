from typing import Any

import torch


class InfiniteSpaceIntegrator:
    """Exact integral for a normalized spatial kernel on the full plane."""

    def integrate(self, kernel: Any, source: Any, domain: Any) -> torch.Tensor:
        del kernel, domain
        if not hasattr(source, "position"):
            raise TypeError("source must provide position")
        return source.position.new_ones(source.position.shape[0])
