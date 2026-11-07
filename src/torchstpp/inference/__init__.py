"""Model fitting and optimization interfaces."""

from ..finite_source.em import (
    FiniteSourceEMResult,
    finite_source_patch_responsibilities,
    smooth_patch_responsibilities,
    update_finite_source_weights,
)
from .em import (
    BackgroundProbabilityUpdate,
    expected_source_offspring,
    update_background_probabilities,
    update_background_probabilities_distributed,
)
from .adaptive_reference import AdaptiveReferenceInference
from .legacy import FortranEMLBFGS, LegacyEMLBFGS, LegacyEMLBFGSConfig
from .reference import ReferenceSelection, TriggerWeightedMagnitudeSelector
from .result import FitResult, OptimizationStep
from ..evaluation import EventIntensity

__all__ = [
    "BackgroundProbabilityUpdate",
    "EventIntensity",
    "AdaptiveReferenceInference",
    "FiniteSourceEMResult",
    "FitResult",
    "FortranEMLBFGS",
    "LegacyEMLBFGS",
    "LegacyEMLBFGSConfig",
    "OptimizationStep",
    "ReferenceSelection",
    "TriggerWeightedMagnitudeSelector",
    "expected_source_offspring",
    "finite_source_patch_responsibilities",
    "smooth_patch_responsibilities",
    "update_background_probabilities",
    "update_background_probabilities_distributed",
    "update_finite_source_weights",
]
