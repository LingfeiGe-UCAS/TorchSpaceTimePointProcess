"""Optional reference-coordinate selection policies for fitted models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

from .em import expected_source_offspring


@dataclass(frozen=True)
class ReferenceSelection:
    """A selected magnitude coordinate and the evidence used to select it."""

    method: str
    raw_weighted_magnitude: float
    selected_magnitude: float
    selected_offset: float
    total_expected_offspring: float

    def as_dict(self) -> dict[str, float | str]:
        return asdict(self)


@dataclass(frozen=True)
class TriggerWeightedMagnitudeSelector:
    """Choose Mref from source magnitudes weighted by expected offspring."""

    minimum_offset: float = 0.3
    maximum_offset: float = 1.5
    round_to: float = 0.1

    def __post_init__(self) -> None:
        if self.minimum_offset < 0.0:
            raise ValueError("minimum_offset must be non-negative")
        if self.maximum_offset < self.minimum_offset:
            raise ValueError("maximum_offset must not be smaller than minimum_offset")
        if self.round_to <= 0.0:
            raise ValueError("round_to must be positive")

    @torch.no_grad()
    def select(
        self,
        model: Any,
        likelihood: Any,
        catalog: Any,
        background_state: Any,
        *,
        target_chunk_size: int,
        source_chunk_size: int,
    ) -> ReferenceSelection:
        local_ranges = None
        local_target_ranges = getattr(likelihood, "local_target_ranges", None)
        if local_target_ranges is not None:
            local_ranges = local_target_ranges(
                catalog.time,
                catalog.time,
                background_source_count=int(background_state.position.shape[0]),
            )

        offspring = None
        error: Exception | None = None
        try:
            offspring = expected_source_offspring(
                model,
                catalog,
                background_state,
                target_chunk_size=target_chunk_size,
                source_chunk_size=source_chunk_size,
                target_ranges=local_ranges,
            )
        except Exception as caught:
            error = caught

        synchronize_failure = getattr(likelihood, "synchronize_failure", None)
        if synchronize_failure is None:
            if error is not None:
                raise error
        else:
            synchronize_failure(
                error, catalog.time, stage="reference-magnitude selection"
            )
        assert offspring is not None

        global_sum = getattr(likelihood, "global_sum_", None)
        if global_sum is not None:
            global_sum(offspring)

        total_weight = offspring.sum()
        if not bool(torch.isfinite(total_weight)) or float(total_weight.item()) <= 0.0:
            raise FloatingPointError(
                "automatic reference selection found no finite triggered-offspring weight"
            )
        completeness = torch.as_tensor(
            catalog.completeness_magnitude,
            dtype=catalog.magnitude.dtype,
            device=catalog.magnitude.device,
        )
        weighted_magnitude = (offspring * catalog.magnitude).sum() / total_weight
        raw_offset = float((weighted_magnitude - completeness).item())
        selected_offset = min(max(raw_offset, self.minimum_offset), self.maximum_offset)
        selected_offset = round(selected_offset / self.round_to) * self.round_to
        selected_offset = min(
            max(selected_offset, self.minimum_offset), self.maximum_offset
        )
        return ReferenceSelection(
            method="trigger_weighted",
            raw_weighted_magnitude=float(weighted_magnitude.item()),
            selected_magnitude=float(completeness.item()) + selected_offset,
            selected_offset=selected_offset,
            total_expected_offspring=float(total_weight.item()),
        )
