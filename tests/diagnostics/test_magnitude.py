import torch

from torchstpp.diagnostics import (
    estimate_completeness_maxc,
    magnitude_frequency_distribution,
)


def test_magnitude_frequency_distribution_counts_both_directions():
    magnitudes = torch.tensor([0.0, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3])

    result = magnitude_frequency_distribution(magnitudes, bin_width=0.1)

    assert torch.allclose(
        result.bin_centers, torch.tensor([0.0, 0.1, 0.2, 0.3], dtype=torch.float64)
    )
    assert torch.equal(result.incremental_counts, torch.tensor([1, 2, 3, 1]))
    assert torch.equal(result.cumulative_counts, torch.tensor([7, 6, 4, 1]))


def test_maxc_reports_raw_peak_and_explicit_correction():
    magnitudes = torch.tensor([-0.1, 0.0, 0.1, 0.1, 0.1, 0.2, 0.3])

    raw = estimate_completeness_maxc(magnitudes, bin_width=0.1)
    corrected = estimate_completeness_maxc(
        magnitudes, bin_width=0.1, correction=0.2
    )

    assert abs(raw.magnitude - 0.1) < 1e-12
    assert raw.peak_count == 3
    assert raw.event_count == 7
    assert corrected.peak_bin_center == raw.peak_bin_center
    assert abs(corrected.magnitude - 0.3) < 1e-12
    assert corrected.event_count_at_or_above == 1


def test_magnitude_diagnostics_reject_invalid_input():
    for values in (torch.tensor([]), torch.tensor([0.0, float("nan")])):
        try:
            estimate_completeness_maxc(values)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid magnitudes must be rejected")
