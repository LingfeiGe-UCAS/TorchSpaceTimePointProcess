from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LikelihoodTerms:
    negative_log_likelihood: torch.Tensor
    log_intensity: torch.Tensor
    background_compensator: torch.Tensor
    triggering_compensator: torch.Tensor
    target_count: int

    @property
    def compensator(self) -> torch.Tensor:
        return self.background_compensator + self.triggering_compensator
