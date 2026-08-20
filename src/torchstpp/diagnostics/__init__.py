"""Catalog diagnostics that can be run before fitting a point-process model."""

from .magnitude import (
    CompletenessEstimate,
    MagnitudeFrequencyDistribution,
    estimate_completeness_maxc,
    magnitude_frequency_distribution,
)

__all__ = [
    "CompletenessEstimate",
    "MagnitudeFrequencyDistribution",
    "estimate_completeness_maxc",
    "magnitude_frequency_distribution",
]
