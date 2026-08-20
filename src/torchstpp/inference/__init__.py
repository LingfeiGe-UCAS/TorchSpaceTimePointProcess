"""Model fitting and optimization interfaces."""

from .em import expected_source_offspring, update_background_probabilities
from .adaptive_reference import AdaptiveReferenceInference
from .legacy import FortranEMLBFGS, LegacyEMLBFGS, LegacyEMLBFGSConfig
from .reference import ReferenceSelection, TriggerWeightedMagnitudeSelector
from .result import FitResult, OptimizationStep

__all__ = [
    "AdaptiveReferenceInference",
    "FitResult",
    "FortranEMLBFGS",
    "LegacyEMLBFGS",
    "LegacyEMLBFGSConfig",
    "OptimizationStep",
    "ReferenceSelection",
    "TriggerWeightedMagnitudeSelector",
    "expected_source_offspring",
    "update_background_probabilities",
]
