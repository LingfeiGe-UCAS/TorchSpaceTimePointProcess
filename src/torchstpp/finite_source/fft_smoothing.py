"""Cached, masked linear FFT smoothing on an axis-aligned horizontal lattice.

Depth uses the original (non-stationary) conditional Beta matrix. No Gaussian
tail truncation, spatial interpolation, or periodic wrap-around is introduced.
Only geometry and fixed smoothing settings are cached, never responsibilities.
"""

from __future__ import annotations

import torch

from ..models.components import conditional_beta_depth_density


def _axis_grid(values: torch.Tensor) -> tuple[torch.Tensor, int, float]:
    unique, inverse = torch.unique(values, sorted=True, return_inverse=True)
    if unique.numel() == 1:
        return torch.zeros_like(inverse), 1, 1.0
    # Missing lattice sites are allowed, but at least one adjacent spacing
    # must be represented. Reject incompatible axes instead of interpolating.
    step = float(torch.diff(unique).min())
    offset = unique - unique[0]
    index = torch.round(offset / step).to(torch.int64)
    float_index = index.to(values.dtype)
    # Refit the spacing to avoid magnifying subtraction round-off across the
    # entire axis. All tests are in coordinate units, relative to input dtype.
    step = float((offset * float_index).sum() / float_index.square().sum())
    residual = (offset - float_index * step).abs().max()
    tolerance = 8 * torch.finfo(values.dtype).eps * max(
        float(unique.abs().max()), step
    )
    if tolerance > 0.01 * step or float(residual) > tolerance:
        raise ValueError(
            "FFT smoothing requires a regular axis-aligned horizontal lattice; "
            "use smoothing_method='blocked' for irregular or unresolved coordinates"
        )
    return index[inverse], int(index[-1]) + 1, step


class FFTSmoothingPlan:
    """One finite-source group's fixed linear smoothing operator.

    Missing horizontal sites and missing depth/site pairs are represented by a
    mask. Input/output ordering follows the original patch bank exactly.
    """

    def __init__(
        self, position, depth, *, horizontal_bandwidth, depth_concentration,
        maximum_depth, boundary_offset, normalization,
    ):
        if position.dtype not in (torch.float32, torch.float64):
            raise ValueError("FFT smoothing supports float32 and float64 only")
        if normalization not in {"fortran", "corrected"}:
            raise ValueError("unknown smoothing normalization")
        if not bool(torch.isfinite(position).all() & torch.isfinite(depth).all()):
            raise ValueError("FFT patch coordinates must be finite")
        # Keep tensor references alive so cached storage identities cannot be
        # reused by an unrelated geometry allocation.
        self.position, self.depth = position, depth
        ix, nx, dx = _axis_grid(position[:, 0])
        iy, ny, dy = _axis_grid(position[:, 1])
        levels, iz = torch.unique(depth, sorted=True, return_inverse=True)
        self.shape = (int(levels.numel()), nx, ny)
        self.index = (iz * nx + ix) * ny + iy
        if torch.unique(self.index).numel() != depth.numel():
            raise ValueError("FFT smoothing does not accept duplicate patch coordinates")
        self.fft_shape = tuple(1 << (2 * n - 2).bit_length() for n in (nx, ny))
        grid_size = self.shape[0] * nx * ny
        padded_size = self.shape[0] * self.fft_shape[0] * self.fft_shape[1]
        if grid_size > 32 * depth.numel() or padded_size > 32_000_000:
            raise ValueError(
                "FFT bounding grid is too sparse or large; use blocked smoothing "
                "or subdivide the geometry"
            )
        ox = torch.arange(1 - nx, nx, device=position.device)
        oy = torch.arange(1 - ny, ny, device=position.device)
        kernel = position.new_zeros(self.fft_shape)
        distance2 = (ox.to(position.dtype)[:, None] * dx).square() + (
            oy.to(position.dtype)[None, :] * dy
        ).square()
        kernel[ox[:, None] % self.fft_shape[0], oy[None, :] % self.fft_shape[1]] = (
            torch.exp(-distance2 / (2 * horizontal_bandwidth**2))
        )
        self.spectrum = torch.fft.rfft2(kernel)
        self.depth_kernel = conditional_beta_depth_density(
            levels[:, None], levels[None, :], maximum_depth,
            position.new_tensor(depth_concentration), boundary_offset=boundary_offset,
        )
        mask = position.new_zeros(grid_size)
        mask[self.index] = 1
        horizontal = self._horizontal(mask.reshape(self.shape))
        # K.T @ mask for corrected, K @ mask for the legacy row-sum mode.
        depth_matrix = self.depth_kernel.T if normalization == "corrected" else self.depth_kernel
        normalizer_grid = depth_matrix @ horizontal.reshape(self.shape[0], -1)
        self.normalizer = normalizer_grid.flatten()[self.index]
        if not bool(torch.isfinite(self.normalizer).all()) or bool((self.normalizer <= 0).any()):
            raise FloatingPointError("FFT smoothing normalizers must be finite and positive")

    def _horizontal(self, grid):
        transformed = torch.fft.rfft2(grid, s=self.fft_shape)
        result = torch.fft.irfft2(transformed * self.spectrum, s=self.fft_shape)
        return result[:, :self.shape[1], :self.shape[2]]

    def apply(self, expected):
        grid = expected.new_zeros(self.shape)
        grid.view(-1)[self.index] = expected / self.normalizer
        horizontal = self._horizontal(grid)
        result = (self.depth_kernel @ horizontal.reshape(self.shape[0], -1)).flatten()[self.index]
        if not bool(torch.isfinite(result).all()):
            raise FloatingPointError("FFT smoothing produced non-finite weights")
        # FFT cancellation can create tiny negative tails. Reject substantial
        # negativity, clip round-off only; do not rescale legacy Fortran mass.
        tolerance = 64 * torch.finfo(result.dtype).eps * result.abs().max()
        if bool((result < -tolerance).any()):
            raise FloatingPointError("FFT smoothing produced negative weights beyond round-off")
        return result.clamp_min(0)
