"""Finite-source subfault responsibility and smoothing updates.

The legacy Fortran program treats the subfault weights (``tau``) as an
event-level latent state rather than as an L-BFGS parameter.  This module
implements the corresponding E-step/rescale operation without coupling it to
the catalog loader or to a particular finite-source model implementation.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch

from ..models.components import conditional_beta_depth_density
from ..models.backgrounds import KDEState
from ..evaluation import EventIntensity
from .fft_smoothing import FFTSmoothingPlan


@dataclass(frozen=True)
class FiniteSourceEMResult:
    """Diagnostics from one finite-source responsibility/rescale update."""

    expected_patch_offspring: torch.Tensor
    patch_tau: torch.Tensor
    smoothing_normalizer: torch.Tensor
    zero_responsibility_groups: tuple[int, ...]


def _finite_source_parents(model: Any, catalog: Any) -> torch.Tensor | None:
    method = getattr(model, "finite_source_parent_indices", None)
    if not callable(method):
        return None
    parents = method(catalog)
    if not isinstance(parents, torch.Tensor):
        raise TypeError("finite_source_parent_indices() must return a tensor")
    if parents.dtype != torch.int64 or parents.ndim != 1:
        raise TypeError(
            "finite_source_parent_indices() must return a one-dimensional "
            "int64 tensor"
        )
    if parents.device != catalog.time.device:
        raise ValueError("finite-source parent indices must use the catalog device")
    if parents.numel() == 0:
        return parents
    event_count = int(catalog.time.numel())
    if bool(torch.any((parents < 0) | (parents >= event_count))):
        raise ValueError("finite-source parent indices lie outside the catalog")
    if torch.unique(parents).numel() != parents.numel():
        raise ValueError("finite-source parent indices must be unique")
    return parents


def _validated_ranges(
    target_ranges: Iterable[tuple[int, int]] | None,
    event_count: int,
) -> tuple[tuple[int, int], ...]:
    ranges = (
        ((0, event_count),)
        if target_ranges is None
        else tuple((int(begin), int(end)) for begin, end in target_ranges)
    )
    occupied: list[tuple[int, int]] = []
    for begin, end in ranges:
        if begin < 0 or end < begin or end > event_count:
            raise ValueError("target ranges must lie inside the catalog")
        if end > begin:
            occupied.append((begin, end))
    occupied.sort()
    if any(right_begin < left_end for (_, left_end), (right_begin, _) in zip(
        occupied, occupied[1:]
    )):
        raise ValueError("target ranges must be disjoint")
    return ranges


def _patch_geometry(model: Any) -> tuple[torch.Tensor, torch.Tensor]:
    geometry = getattr(model, "geometry", None)
    if geometry is None:
        geometry = getattr(model, "finite_source_geometry", None)
    if geometry is None:
        raise TypeError("finite-source model must expose geometry")
    position = getattr(geometry, "position", None)
    depth = getattr(geometry, "depth", None)
    if not isinstance(position, torch.Tensor) or not isinstance(depth, torch.Tensor):
        raise TypeError("finite-source geometry must expose position and depth tensors")
    patch_count = int(position.shape[0]) if position.ndim == 2 else -1
    if patch_count < 1 or position.shape[1:] != (2,):
        raise ValueError("finite-source geometry.position must have shape [P, 2]")
    if depth.shape != (patch_count,):
        raise ValueError("finite-source geometry.depth must have shape [P]")
    if position.dtype != depth.dtype or position.device != depth.device:
        raise ValueError("finite-source patch position and depth must share dtype/device")
    return position, depth


def _patch_group(
    model: Any,
    catalog: Any,
    parent_index: int,
    patch_count: int,
) -> tuple[int, int, int, Any]:
    finite_source_id = getattr(catalog, "finite_source_id", None)
    if not isinstance(finite_source_id, torch.Tensor):
        raise TypeError("finite-source catalog must provide finite_source_id")
    group = int(finite_source_id[parent_index].item())
    if group <= 0:
        raise ValueError("finite-source parents must have a positive finite_source_id")
    patch_slice = getattr(model, "patch_slice", None)
    if not callable(patch_slice):
        raise TypeError("finite-source model must provide patch_slice(group)")
    group_slice = patch_slice(group)
    if not isinstance(group_slice, slice):
        raise TypeError("patch_slice(group) must return a slice")
    if group_slice.step not in (None, 1):
        raise ValueError("finite-source patch slices must be contiguous")
    begin = 0 if group_slice.start is None else int(group_slice.start)
    end = patch_count if group_slice.stop is None else int(group_slice.stop)
    if begin < 0 or end <= begin or end > patch_count:
        raise ValueError("finite-source patch slice lies outside geometry")
    parent = catalog.take(slice(parent_index, parent_index + 1))
    return group, begin, end, parent


def _effective_settings(
    model: Any,
    *,
    patch_chunk_size: int | None,
    horizontal_smoothing_bandwidth: float | None,
    depth_smoothing_concentration: float | None,
    tau_normalization: str | None,
    zero_responsibility_policy: str | None,
) -> tuple[int, float, float, str, str]:
    patch_chunk_size = int(
        getattr(model, "patch_chunk_size", 4096)
        if patch_chunk_size is None
        else patch_chunk_size
    )
    horizontal_smoothing_bandwidth = float(
        getattr(model, "horizontal_smoothing_bandwidth", float("nan"))
        if horizontal_smoothing_bandwidth is None
        else horizontal_smoothing_bandwidth
    )
    depth_smoothing_concentration = float(
        getattr(model, "depth_smoothing_concentration", float("nan"))
        if depth_smoothing_concentration is None
        else depth_smoothing_concentration
    )
    tau_normalization = str(
        getattr(model, "tau_normalization", "corrected")
        if tau_normalization is None
        else tau_normalization
    ).lower()
    zero_responsibility_policy = str(
        getattr(model, "zero_responsibility_policy", "retain")
        if zero_responsibility_policy is None
        else zero_responsibility_policy
    ).lower()
    if zero_responsibility_policy == "keep":
        zero_responsibility_policy = "retain"
    if patch_chunk_size < 1:
        raise ValueError("patch_chunk_size must be positive")
    if not torch.isfinite(torch.tensor(horizontal_smoothing_bandwidth)):
        raise ValueError("horizontal_smoothing_bandwidth must be finite")
    if horizontal_smoothing_bandwidth <= 0.0:
        raise ValueError("horizontal_smoothing_bandwidth must be positive")
    if not torch.isfinite(torch.tensor(depth_smoothing_concentration)):
        raise ValueError("depth_smoothing_concentration must be finite")
    if depth_smoothing_concentration < 0.0:
        raise ValueError("depth_smoothing_concentration must be non-negative")
    if tau_normalization not in {"fortran", "corrected"}:
        raise ValueError("tau_normalization must be 'fortran' or 'corrected'")
    if zero_responsibility_policy not in {"retain", "uniform"}:
        raise ValueError(
            "zero_responsibility_policy must be 'retain' or 'uniform'"
        )
    return (
        patch_chunk_size,
        horizontal_smoothing_bandwidth,
        depth_smoothing_concentration,
        tau_normalization,
        zero_responsibility_policy,
    )


def _event_intensity(
    model: Any,
    catalog: Any,
    target: Any,
    state: KDEState,
    *,
    source_chunk_size: int,
) -> torch.Tensor:
    background = model.background_rate(model.background.evaluate(target, state))
    intensity = background.clone()
    event_count = int(catalog.time.numel())
    for source_begin in range(0, event_count, source_chunk_size):
        source_end = min(source_begin + source_chunk_size, event_count)
        source = catalog.take(slice(source_begin, source_end))
        if source.time[0] > target.time[-1]:
            break
        intensity.add_(model.pairwise_rate(target, source).sum(dim=-1))
    if not bool(torch.all(torch.isfinite(intensity))) or bool(
        torch.any(intensity <= 0.0)
    ):
        raise FloatingPointError("event intensity must be finite and positive")
    return intensity


@torch.no_grad()
def finite_source_patch_responsibilities(
    model: Any,
    catalog: Any,
    state: KDEState,
    *,
    target_chunk_size: int = 2048,
    source_chunk_size: int = 4096,
    patch_chunk_size: int | None = None,
    target_ranges: Iterable[tuple[int, int]] | None = None,
    event_intensity: EventIntensity | None = None,
) -> torch.Tensor | None:
    """Sum direct-offspring responsibilities for every finite-source patch.

    The returned value is the Fortran ``tau_s`` quantity.  Patch rates must
    include the current normalized ``tau`` weight, so summing a finite event's
    patches recovers its ordinary event-level triggering contribution.
    """

    parents = _finite_source_parents(model, catalog)
    if parents is None or parents.numel() == 0:
        return None
    state.validate()
    if target_chunk_size < 1 or source_chunk_size < 1:
        raise ValueError("target and source chunk sizes must be positive")
    patch_tau = getattr(model, "patch_tau", None)
    if not isinstance(patch_tau, torch.Tensor) or patch_tau.ndim != 1:
        raise TypeError("finite-source model must expose one-dimensional patch_tau")
    position, _ = _patch_geometry(model)
    patch_count = int(position.shape[0])
    if patch_tau.shape != (patch_count,):
        raise ValueError("patch_tau and finite-source geometry sizes differ")
    if patch_tau.device != catalog.time.device or patch_tau.dtype != catalog.time.dtype:
        raise ValueError("patch_tau must share the catalog dtype and device")
    effective_patch_chunk = int(
        getattr(model, "patch_chunk_size", source_chunk_size)
        if patch_chunk_size is None
        else patch_chunk_size
    )
    if effective_patch_chunk < 1:
        raise ValueError("patch_chunk_size must be positive")

    event_count = int(catalog.time.numel())
    ranges = _validated_ranges(target_ranges, event_count)
    if event_intensity is not None:
        event_intensity.validate_for(model, catalog, state, ranges)
    groups = [
        _patch_group(model, catalog, int(index.item()), patch_count)
        for index in parents
    ]
    group_ids = [group for group, _, _, _ in groups]
    if len(set(group_ids)) != len(group_ids):
        raise ValueError("each finite_source_id must identify exactly one parent event")

    expected = torch.zeros_like(patch_tau)
    finite_patch_rate = getattr(model, "finite_patch_rate", None)
    if not callable(finite_patch_rate):
        raise TypeError(
            "finite-source model must provide finite_patch_rate(target, parent, begin, end)"
        )
    for range_begin, range_end in ranges:
        for target_begin in range(range_begin, range_end, target_chunk_size):
            target_end = min(target_begin + target_chunk_size, range_end)
            target = catalog.take(slice(target_begin, target_end))
            intensity = event_intensity.values[target_begin:target_end] if event_intensity is not None else _event_intensity(
                model,
                catalog,
                target,
                state,
                source_chunk_size=source_chunk_size,
            )
            for _, group_begin, group_end, parent in groups:
                if parent.time[0] > target.time[-1]:
                    continue
                for patch_begin in range(
                    group_begin, group_end, effective_patch_chunk
                ):
                    patch_end = min(
                        patch_begin + effective_patch_chunk, group_end
                    )
                    rate = finite_patch_rate(
                        target, parent, patch_begin, patch_end
                    )
                    expected_shape = (
                        target_end - target_begin,
                        patch_end - patch_begin,
                    )
                    if rate.shape != expected_shape:
                        raise ValueError(
                            "finite_patch_rate returned shape "
                            f"{tuple(rate.shape)}, expected {expected_shape}"
                        )
                    if not bool(torch.all(torch.isfinite(rate))) or bool(
                        torch.any(rate < 0.0)
                    ):
                        raise FloatingPointError(
                            "finite-source patch rates must be finite and non-negative"
                        )
                    expected[patch_begin:patch_end].add_(
                        (rate / intensity[:, None]).sum(dim=0)
                    )
    return expected


def _smoothing_kernel(
    target_position: torch.Tensor,
    target_depth: torch.Tensor,
    source_position: torch.Tensor,
    source_depth: torch.Tensor,
    *,
    horizontal_bandwidth: float,
    depth_concentration: float,
    maximum_depth: float,
    boundary_offset: float,
) -> torch.Tensor:
    displacement = target_position[:, None, :] - source_position[None, :, :]
    distance_squared = displacement.square().sum(dim=-1)
    horizontal = torch.exp(
        -distance_squared / (2.0 * horizontal_bandwidth * horizontal_bandwidth)
    )
    concentration = target_position.new_tensor(depth_concentration)
    depth = conditional_beta_depth_density(
        target_depth[:, None],
        source_depth[None, :],
        maximum_depth,
        concentration,
        boundary_offset=boundary_offset,
    )
    return horizontal * depth


def _normalizer_signature(
    position: torch.Tensor,
    depth: torch.Tensor,
    groups: list[tuple[int, int, int, Any]],
    *,
    horizontal_bandwidth: float,
    depth_concentration: float,
    maximum_depth: float,
    boundary_offset: float,
    normalization: str,
) -> tuple[Any, ...]:
    return (
        normalization,
        horizontal_bandwidth,
        depth_concentration,
        maximum_depth,
        boundary_offset,
        str(position.dtype),
        str(position.device),
        position.data_ptr(),
        position._version,
        depth.data_ptr(),
        depth._version,
        tuple((group, begin, end) for group, begin, end, _ in groups),
    )


def _smoothing_normalizer(
    model: Any,
    position: torch.Tensor,
    depth: torch.Tensor,
    groups: list[tuple[int, int, int, Any]],
    *,
    patch_chunk_size: int,
    horizontal_bandwidth: float,
    depth_concentration: float,
    normalization: str,
) -> torch.Tensor:
    maximum_depth = float(getattr(model, "maximum_depth"))
    boundary_offset = float(getattr(model, "depth_boundary_offset", 0.5))
    signature = _normalizer_signature(
        position,
        depth,
        groups,
        horizontal_bandwidth=horizontal_bandwidth,
        depth_concentration=depth_concentration,
        maximum_depth=maximum_depth,
        boundary_offset=boundary_offset,
        normalization=normalization,
    )
    buffer = getattr(model, "patch_smoothing_normalizer", None)
    if not isinstance(buffer, torch.Tensor) or buffer.shape != depth.shape:
        raise TypeError(
            "finite-source model must expose patch_smoothing_normalizer with shape [P]"
        )
    if getattr(model, "_finite_source_normalizer_signature", None) == signature:
        if bool(torch.all(torch.isfinite(buffer))) and bool(torch.all(buffer > 0.0)):
            return buffer

    normalizer = torch.zeros_like(depth)
    for _, group_begin, group_end, _ in groups:
        if normalization == "corrected":
            # Column sums: D_k = sum_l K(l | k).  This is the normalizer
            # required to redistribute each source patch's expected offspring.
            for source_begin in range(
                group_begin, group_end, patch_chunk_size
            ):
                source_end = min(source_begin + patch_chunk_size, group_end)
                value = depth.new_zeros(source_end - source_begin)
                for target_begin in range(
                    group_begin, group_end, patch_chunk_size
                ):
                    target_end = min(target_begin + patch_chunk_size, group_end)
                    kernel = _smoothing_kernel(
                        position[target_begin:target_end],
                        depth[target_begin:target_end],
                        position[source_begin:source_end],
                        depth[source_begin:source_end],
                        horizontal_bandwidth=horizontal_bandwidth,
                        depth_concentration=depth_concentration,
                        maximum_depth=maximum_depth,
                        boundary_offset=boundary_offset,
                    )
                    value.add_(kernel.sum(dim=0))
                normalizer[source_begin:source_end] = value
        else:
            # The Fortran Ite0_tau loop evaluates K(k | m) while rescale later
            # uses K(l | k).  Retain that row-sum convention as an explicit
            # compatibility mode; it differs for the asymmetric depth kernel.
            for target_begin in range(
                group_begin, group_end, patch_chunk_size
            ):
                target_end = min(target_begin + patch_chunk_size, group_end)
                value = depth.new_zeros(target_end - target_begin)
                for source_begin in range(
                    group_begin, group_end, patch_chunk_size
                ):
                    source_end = min(source_begin + patch_chunk_size, group_end)
                    kernel = _smoothing_kernel(
                        position[target_begin:target_end],
                        depth[target_begin:target_end],
                        position[source_begin:source_end],
                        depth[source_begin:source_end],
                        horizontal_bandwidth=horizontal_bandwidth,
                        depth_concentration=depth_concentration,
                        maximum_depth=maximum_depth,
                        boundary_offset=boundary_offset,
                    )
                    value.add_(kernel.sum(dim=1))
                normalizer[target_begin:target_end] = value

    if not bool(torch.all(torch.isfinite(normalizer))) or bool(
        torch.any(normalizer <= 0.0)
    ):
        raise FloatingPointError(
            "finite-source smoothing normalizers must be finite and positive"
        )
    setter = getattr(model, "set_patch_smoothing_normalizer_", None)
    if callable(setter):
        setter(normalizer)
    else:
        buffer.copy_(normalizer)
    setattr(model, "_finite_source_normalizer_signature", signature)
    return buffer


@torch.no_grad()
def smooth_patch_responsibilities(
    model: Any,
    catalog: Any,
    expected: torch.Tensor,
    *,
    patch_chunk_size: int | None = None,
    horizontal_smoothing_bandwidth: float | None = None,
    depth_smoothing_concentration: float | None = None,
    tau_normalization: str | None = None,
    zero_responsibility_policy: str | None = None,
    smoothing_method: str | None = None,
) -> FiniteSourceEMResult | None:
    """Smooth patch offspring counts and update the model's ``patch_tau``."""

    parents = _finite_source_parents(model, catalog)
    if parents is None or parents.numel() == 0:
        return None
    patch_tau = getattr(model, "patch_tau", None)
    if not isinstance(patch_tau, torch.Tensor) or patch_tau.ndim != 1:
        raise TypeError("finite-source model must expose one-dimensional patch_tau")
    position, depth = _patch_geometry(model)
    patch_count = int(position.shape[0])
    if expected.shape != (patch_count,):
        raise ValueError("expected patch offspring must have shape [P]")
    if expected.dtype != patch_tau.dtype or expected.device != patch_tau.device:
        raise ValueError("expected patch offspring must share patch_tau dtype/device")
    if not bool(torch.all(torch.isfinite(expected))) or bool(torch.any(expected < 0.0)):
        raise FloatingPointError(
            "expected patch offspring must be finite and non-negative"
        )
    (
        effective_patch_chunk,
        horizontal_bandwidth,
        depth_concentration,
        normalization,
        zero_policy,
    ) = _effective_settings(
        model,
        patch_chunk_size=patch_chunk_size,
        horizontal_smoothing_bandwidth=horizontal_smoothing_bandwidth,
        depth_smoothing_concentration=depth_smoothing_concentration,
        tau_normalization=tau_normalization,
        zero_responsibility_policy=zero_responsibility_policy,
    )
    groups = [
        _patch_group(model, catalog, int(index.item()), patch_count)
        for index in parents
    ]
    group_ids = [group for group, _, _, _ in groups]
    if len(set(group_ids)) != len(group_ids):
        raise ValueError("each finite_source_id must identify exactly one parent event")
    method = str(smoothing_method or getattr(model, "smoothing_method", "blocked")).lower()
    if method not in {"blocked", "fft"}:
        raise ValueError("smoothing_method must be 'blocked' or 'fft'")
    plans = None
    if method == "fft":
        signature = _normalizer_signature(
            position, depth, groups, horizontal_bandwidth=horizontal_bandwidth,
            depth_concentration=depth_concentration,
            maximum_depth=float(model.maximum_depth),
            boundary_offset=float(getattr(model, "depth_boundary_offset", 0.5)),
            normalization=normalization,
        )
        cached = getattr(model, "_fft_smoothing_cache", None)
        if cached is None or cached[0] != signature:
            plans = [FFTSmoothingPlan(
                position[begin:end], depth[begin:end],
                horizontal_bandwidth=horizontal_bandwidth,
                depth_concentration=depth_concentration,
                maximum_depth=float(model.maximum_depth),
                boundary_offset=float(getattr(model, "depth_boundary_offset", 0.5)),
                normalization=normalization,
            ) for _, begin, end, _ in groups]
            model._fft_smoothing_cache = (signature, plans)
        else:
            plans = cached[1]
        normalizer = model.patch_smoothing_normalizer.clone()
        for plan, (_, begin, end, _) in zip(plans, groups):
            normalizer[begin:end] = plan.normalizer
        model.patch_smoothing_normalizer.copy_(normalizer)
        # The shared buffer now contains FFT values, not the blocked cache.
        model._finite_source_normalizer_signature = None
    else:
        normalizer = _smoothing_normalizer(
            model,
            position,
            depth,
            groups,
            patch_chunk_size=effective_patch_chunk,
            horizontal_bandwidth=horizontal_bandwidth,
            depth_concentration=depth_concentration,
            normalization=normalization,
        )

    maximum_depth = float(getattr(model, "maximum_depth"))
    boundary_offset = float(getattr(model, "depth_boundary_offset", 0.5))
    new_tau = patch_tau.detach().clone()
    zero_groups: list[int] = []
    for ordinal, (group, group_begin, group_end, _) in enumerate(groups):
        group_expected = expected[group_begin:group_end]
        if float(group_expected.sum().item()) <= 0.0:
            zero_groups.append(group)
            if zero_policy == "uniform":
                new_tau[group_begin:group_end].fill_(1.0)
            continue
        scaled = group_expected / normalizer[group_begin:group_end]
        if plans is not None:
            new_tau[group_begin:group_end] = plans[ordinal].apply(group_expected)
            continue
        for target_begin in range(
            group_begin, group_end, effective_patch_chunk
        ):
            target_end = min(target_begin + effective_patch_chunk, group_end)
            value = depth.new_zeros(target_end - target_begin)
            for source_begin in range(
                group_begin, group_end, effective_patch_chunk
            ):
                source_end = min(source_begin + effective_patch_chunk, group_end)
                kernel = _smoothing_kernel(
                    position[target_begin:target_end],
                    depth[target_begin:target_end],
                    position[source_begin:source_end],
                    depth[source_begin:source_end],
                    horizontal_bandwidth=horizontal_bandwidth,
                    depth_concentration=depth_concentration,
                    maximum_depth=maximum_depth,
                    boundary_offset=boundary_offset,
                )
                value.add_(
                    kernel @ scaled[
                        source_begin - group_begin : source_end - group_begin
                    ]
                )
            new_tau[target_begin:target_end] = value
        group_tau = new_tau[group_begin:group_end]
        if not bool(torch.all(torch.isfinite(group_tau))) or bool(
            torch.any(group_tau < 0.0)
        ):
            raise FloatingPointError(
                f"smoothed patch_tau for group {group} is invalid"
            )
        if float(group_tau.sum().item()) <= 0.0:
            zero_groups.append(group)
            if zero_policy == "retain":
                new_tau[group_begin:group_end] = patch_tau[group_begin:group_end]
            else:
                new_tau[group_begin:group_end].fill_(1.0)

    setter = getattr(model, "set_patch_tau_", None)
    if not callable(setter):
        setter = getattr(model, "update_patch_tau_", None)
    if not callable(setter):
        raise TypeError("finite-source model must provide set_patch_tau_(tau)")
    setter(new_tau)
    return FiniteSourceEMResult(
        expected_patch_offspring=expected.detach().clone(),
        patch_tau=patch_tau.detach().clone(),
        smoothing_normalizer=normalizer.detach().clone(),
        zero_responsibility_groups=tuple(dict.fromkeys(zero_groups)),
    )


@torch.no_grad()
def update_finite_source_weights(
    model: Any,
    catalog: Any,
    state: KDEState,
    *,
    target_chunk_size: int = 2048,
    source_chunk_size: int = 4096,
    patch_chunk_size: int | None = None,
    target_ranges: Iterable[tuple[int, int]] | None = None,
    distributed_likelihood: Any | None = None,
    horizontal_smoothing_bandwidth: float | None = None,
    depth_smoothing_concentration: float | None = None,
    tau_normalization: str | None = None,
    zero_responsibility_policy: str | None = None,
    event_intensity: EventIntensity | None = None,
) -> FiniteSourceEMResult | None:
    """Run one finite-source E-step and update replicated patch weights.

    In distributed use, targets are disjoint across ranks while patch geometry is
    replicated.  A single all-reduce of the length-``P`` responsibility vector
    reconstructs the exact global update before every rank performs the same
    deterministic smoothing operation.
    """

    parents = _finite_source_parents(model, catalog)
    if parents is None or parents.numel() == 0:
        return None

    global_sum = None
    synchronize_failure = None
    if distributed_likelihood is not None:
        global_sum = getattr(distributed_likelihood, "global_sum_", None)
        synchronize_failure = getattr(
            distributed_likelihood, "synchronize_failure", None
        )
        if not callable(global_sum):
            raise TypeError("distributed_likelihood must provide global_sum_()")
        if target_ranges is None:
            schedule = getattr(distributed_likelihood, "local_target_schedule", None)
            ranges_method = getattr(
                distributed_likelihood, "local_target_ranges", None
            )
            if callable(schedule):
                target_ranges, target_chunk_size, source_chunk_size = schedule(
                    catalog.time,
                    catalog.time,
                    background_source_count=int(state.position.shape[0]),
                )
            elif callable(ranges_method):
                target_ranges = ranges_method(
                    catalog.time,
                    catalog.time,
                    background_source_count=int(state.position.shape[0]),
                )
            else:
                raise TypeError(
                    "distributed_likelihood must provide local target scheduling"
                )

    expected: torch.Tensor | None = None
    local_error: Exception | None = None
    try:
        expected = finite_source_patch_responsibilities(
            model,
            catalog,
            state,
            target_chunk_size=target_chunk_size,
            source_chunk_size=source_chunk_size,
            patch_chunk_size=patch_chunk_size,
            target_ranges=target_ranges,
            event_intensity=event_intensity,
        )
    except Exception as caught:
        local_error = caught
    if callable(synchronize_failure):
        synchronize_failure(
            local_error, catalog.time, stage="finite-source E-step"
        )
    elif local_error is not None:
        raise local_error
    assert expected is not None

    if callable(global_sum):
        global_sum(expected)
    reduction_error: Exception | None = None
    result: FiniteSourceEMResult | None = None
    try:
        if not bool(torch.all(torch.isfinite(expected))) or bool(
            torch.any(expected < 0.0)
        ):
            raise FloatingPointError(
                "global finite-source responsibilities are invalid"
            )
        result = smooth_patch_responsibilities(
            model,
            catalog,
            expected,
            patch_chunk_size=patch_chunk_size,
            horizontal_smoothing_bandwidth=horizontal_smoothing_bandwidth,
            depth_smoothing_concentration=depth_smoothing_concentration,
            tau_normalization=tau_normalization,
            zero_responsibility_policy=zero_responsibility_policy,
        )
    except Exception as caught:
        reduction_error = caught
    if callable(synchronize_failure):
        synchronize_failure(
            reduction_error,
            catalog.time,
            stage="finite-source rescale",
        )
    elif reduction_error is not None:
        raise reduction_error
    assert result is not None
    return result


__all__ = [
    "FiniteSourceEMResult",
    "finite_source_patch_responsibilities",
    "smooth_patch_responsibilities",
    "update_finite_source_weights",
]
