from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RelationBlock:
    """One thresholded block of causal event relations stored on CPU."""

    source_index: torch.Tensor
    target_index: torch.Tensor
    trigger_rate: torch.Tensor
    trigger_probability: torch.Tensor

    @property
    def row_count(self) -> int:
        return self.source_index.shape[0]
