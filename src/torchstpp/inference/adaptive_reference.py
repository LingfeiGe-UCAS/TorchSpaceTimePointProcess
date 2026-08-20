"""Composable inference wrapper for data-adaptive model coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .reference import ReferenceSelection, TriggerWeightedMagnitudeSelector
from .result import FitResult


@dataclass
class AdaptiveReferenceInference:
    """Run a pilot fit, choose a reference coordinate, then run the main fit."""

    pilot: Any
    base: Any
    selector: TriggerWeightedMagnitudeSelector
    pilot_outer_iterations: int

    def __post_init__(self) -> None:
        if self.pilot_outer_iterations < 1:
            raise ValueError("pilot_outer_iterations must be positive")
        self.last_selection: ReferenceSelection | None = None

    def fit(
        self,
        model: Any,
        likelihood: Any,
        catalog: Any,
        domain: Any,
        background_state: Any,
    ) -> FitResult:
        recenter = getattr(model, "recenter_depth_reference_magnitude_offset", None)
        if recenter is None:
            raise TypeError(
                "adaptive reference inference requires a model supporting "
                "recenter_depth_reference_magnitude_offset"
            )
        pilot_result = self.pilot.fit(
            model, likelihood, catalog, domain, background_state
        )
        selection = self.selector.select(
            model,
            likelihood,
            catalog,
            pilot_result.background_state,
            target_chunk_size=self.base.config.target_chunk_size,
            source_chunk_size=self.base.config.source_chunk_size,
        )
        recenter(selection.selected_offset)
        broadcast_parameters = getattr(likelihood, "broadcast_parameters_", None)
        if broadcast_parameters is not None:
            broadcast_parameters(model)
        metadata = selection.as_dict()
        metadata["pilot_outer_iterations"] = self.pilot_outer_iterations
        model.depth_reference_selection = metadata
        self.last_selection = selection
        if getattr(likelihood, "is_main_process", True):
            print(
                "automatic depth reference: "
                f"raw Mref={selection.raw_weighted_magnitude:.6g}, "
                f"selected Mref={selection.selected_magnitude:.6g}, "
                f"offset={selection.selected_offset:.6g}, "
                f"expected_offspring={selection.total_expected_offspring:.6g}",
                flush=True,
            )
        return self.base.fit(
            model,
            likelihood,
            catalog,
            domain,
            pilot_result.background_state,
        )
