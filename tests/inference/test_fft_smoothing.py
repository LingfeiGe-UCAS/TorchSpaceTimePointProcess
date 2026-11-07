"""Compare FFT with the full discrete operator, including asymmetric depth."""
from pathlib import Path

import pytest
import torch

from torchstpp.finite_source.em import _smoothing_kernel, smooth_patch_responsibilities
from torchstpp.finite_source.fft_smoothing import FFTSmoothingPlan
from torchstpp.config import TorchSTPPConfig, build_model
from torchstpp.finite_source import FiniteSourceGeometry
from tests.inference.test_finite_source_em import _problem, _ConstantBackground


@pytest.mark.parametrize("normalization", ["fortran", "corrected"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_masked_shuffled_grid_matches_dense(normalization, dtype, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    points = torch.cartesian_prod(
        torch.arange(4), torch.arange(3), torch.arange(3)
    ).to(device=device, dtype=dtype)
    # Asymmetric mask also removes selected depth/site pairs.
    points = points[torch.arange(len(points), device=device) % 5 != 0].flip(0)
    position = points[:, :2] * 0.1
    depth = points[:, 2] * 15  # Includes both depth boundaries.
    settings = dict(horizontal_bandwidth=0.13, depth_concentration=7.0,
                    maximum_depth=30.0, boundary_offset=0.5)
    kernel = _smoothing_kernel(position, depth, position, depth, **settings)
    normalizer = kernel.sum(0 if normalization == "corrected" else 1)
    plan = FFTSmoothingPlan(position, depth, normalization=normalization, **settings)
    expected = torch.arange(1, len(depth) + 1, device=device, dtype=dtype).square()
    for values in (expected, torch.nn.functional.one_hot(
        torch.tensor(0, device=device), len(depth)
    ).to(dtype), torch.zeros_like(expected)):
        reference = kernel @ (values / normalizer)
        result = plan.apply(values)
        tolerance = 3e-5 if dtype == torch.float32 else 2e-12
        torch.testing.assert_close(plan.normalizer, normalizer, rtol=tolerance, atol=tolerance)
        torch.testing.assert_close(result, reference, rtol=tolerance, atol=tolerance * max(1, float(reference.max())))
        if normalization == "corrected":
            torch.testing.assert_close(result.sum(), values.sum(), rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("position", [
    [[0., 0.], [.1, 0.], [.23, 0.]],
    [[0., 0.], [0., 0.]],
])
def test_rejects_irregular_and_duplicate_points(position):
    position = torch.tensor(position, dtype=torch.float64)
    with pytest.raises(ValueError):
        FFTSmoothingPlan(position, position.new_ones(len(position)),
                         horizontal_bandwidth=.1, depth_concentration=10.,
                         maximum_depth=30., boundary_offset=.5, normalization="corrected")


def test_public_dispatch_cache_switch_invalidation_and_zero_policy():
    model, catalog, _ = _problem(
        patch_position=torch.tensor([[0., 0.], [.1, 0.]], dtype=torch.float64),
        patch_depth=torch.tensor([2., 24.], dtype=torch.float64),
    )
    values = torch.tensor([2., 7.], dtype=torch.float64)
    reference = smooth_patch_responsibilities(model, catalog, values)
    model.smoothing_method = "fft"
    actual = smooth_patch_responsibilities(model, catalog, values)
    torch.testing.assert_close(actual.patch_tau, reference.patch_tau)
    cache = model._fft_smoothing_cache
    smooth_patch_responsibilities(model, catalog, values)
    assert model._fft_smoothing_cache is cache
    model.geometry.position.mul_(2)
    actual = smooth_patch_responsibilities(model, catalog, values)
    assert model._fft_smoothing_cache is not cache
    reference = smooth_patch_responsibilities(model, catalog, values, smoothing_method="blocked")
    torch.testing.assert_close(actual.patch_tau, reference.patch_tau)
    zero = smooth_patch_responsibilities(model, catalog, values * 0)
    torch.testing.assert_close(zero.patch_tau, actual.patch_tau)
    model.zero_responsibility_policy = "uniform"
    zero = smooth_patch_responsibilities(model, catalog, values * 0)
    torch.testing.assert_close(zero.patch_tau, torch.ones_like(values))


def test_single_patch_identity():
    position = torch.tensor([[3., 5.]], dtype=torch.float64)
    plan = FFTSmoothingPlan(position, position.new_tensor([8.]),
                            horizontal_bandwidth=.1, depth_concentration=100.,
                            maximum_depth=30., boundary_offset=.5, normalization="corrected")
    torch.testing.assert_close(plan.apply(position.new_tensor([7.])), position.new_tensor([7.]))


@pytest.mark.parametrize("method", ["blocked", "fft", "invalid"])
def test_builder_passes_method_and_rejects_invalid(method):
    geometry = FiniteSourceGeometry(
        patch_id=torch.tensor([1]), finite_source_id=torch.tensor([1]),
        position=torch.zeros((1, 2), dtype=torch.float64),
        depth=torch.tensor([10.], dtype=torch.float64),
    )
    config = TorchSTPPConfig(
        {"model": {"type": "finite_source_etas3d", "maximum_depth": 30.},
         "finite_source": {"smoothing_method": method}}, Path("test.toml"),
    )
    if method == "invalid":
        with pytest.raises(ValueError, match="smoothing_method"):
            build_model(config, background=_ConstantBackground(), finite_source_geometry=geometry)
    else:
        model = build_model(config, background=_ConstantBackground(), finite_source_geometry=geometry)
        assert model.smoothing_method == method
