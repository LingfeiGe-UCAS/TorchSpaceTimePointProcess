"""Magnitude-frequency diagnostics and simple completeness estimators."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MagnitudeFrequencyDistribution:
    """Incremental and reverse-cumulative event counts on magnitude bins."""

    bin_centers: torch.Tensor
    incremental_counts: torch.Tensor
    cumulative_counts: torch.Tensor
    bin_width: float


@dataclass(frozen=True)
class CompletenessEstimate:
    """One magnitude-of-completeness estimate and its basic audit values."""

    method: str
    magnitude: float
    bin_width: float
    correction: float
    peak_bin_center: float
    peak_count: int
    event_count: int
    event_count_at_or_above: int


def _validated_magnitudes(magnitudes: torch.Tensor) -> torch.Tensor:
    values = torch.as_tensor(magnitudes)
    if values.ndim != 1:
        raise ValueError("magnitudes must be a one-dimensional tensor")
    if values.numel() == 0:
        raise ValueError("magnitudes must not be empty")
    if not values.is_floating_point():
        values = values.to(torch.get_default_dtype())
    if not bool(torch.all(torch.isfinite(values))):
        raise ValueError("magnitudes contain NaN or infinite values")
    return values


def magnitude_frequency_distribution(
    magnitudes: torch.Tensor,
    *,
    bin_width: float = 0.1,
) -> MagnitudeFrequencyDistribution:
    """Bin magnitudes and return incremental and ``M >= bin`` counts.

    Bin centers are integer multiples of ``bin_width``. Computation uses
    float64 for stable bin assignment even when the input catalog is float32.
    """

    values = _validated_magnitudes(magnitudes)
    if not torch.isfinite(torch.tensor(bin_width)) or bin_width <= 0.0:
        raise ValueError("bin_width must be a positive finite number")

    work = values.to(dtype=torch.float64)
    width = work.new_tensor(bin_width)
    bin_numbers = torch.floor(work / width + 0.5).to(torch.int64)
    first = int(bin_numbers.amin().item())
    last = int(bin_numbers.amax().item())
    incremental = torch.bincount(
        bin_numbers - first,
        minlength=last - first + 1,
    )
    centers = torch.arange(
        first,
        last + 1,
        dtype=torch.float64,
        device=values.device,
    ) * width
    cumulative = torch.flip(
        torch.cumsum(torch.flip(incremental, dims=(0,)), dim=0),
        dims=(0,),
    )
    return MagnitudeFrequencyDistribution(
        bin_centers=centers,
        incremental_counts=incremental,
        cumulative_counts=cumulative,
        bin_width=float(bin_width),
    )


def estimate_completeness_maxc(
    magnitudes: torch.Tensor,
    *,
    bin_width: float = 0.1,
    correction: float = 0.0,
) -> CompletenessEstimate:
    """Estimate completeness from the peak incremental magnitude bin.

    ``correction`` is explicit because some workflows report raw MAXC while
    others add a fixed offset (often 0.2 magnitude units). The estimator does
    not silently choose between those scientific conventions.
    """

    values = _validated_magnitudes(magnitudes)
    if not torch.isfinite(torch.tensor(correction)):
        raise ValueError("correction must be finite")
    distribution = magnitude_frequency_distribution(values, bin_width=bin_width)
    peak_index = int(torch.argmax(distribution.incremental_counts).item())
    peak_center = float(distribution.bin_centers[peak_index].item())
    completeness = peak_center + float(correction)
    count_above = int(torch.count_nonzero(values >= completeness).item())
    return CompletenessEstimate(
        method="maximum_curvature",
        magnitude=completeness,
        bin_width=float(bin_width),
        correction=float(correction),
        peak_bin_center=peak_center,
        peak_count=int(distribution.incremental_counts[peak_index].item()),
        event_count=int(values.numel()),
        event_count_at_or_above=count_above,
    )
