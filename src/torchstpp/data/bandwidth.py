"""Memory-bounded adaptive bandwidth estimators."""

from __future__ import annotations

import math

import torch


@torch.no_grad()
def knn_bandwidth(
    position: torch.Tensor,
    *,
    neighbors: int = 5,
    minimum: float = 0.05,
    target_chunk_size: int = 2048,
    source_chunk_size: int = 4096,
) -> torch.Tensor:
    """Return the legacy bandwidth: max(k-th neighbor distance, minimum).

    Self-distance is excluded. Distances are evaluated in two-dimensional
    blocks and only the current k smallest values are retained, so no complete
    ``N x N`` distance matrix is materialized.
    """

    if not isinstance(position, torch.Tensor):
        raise TypeError("position must be a torch.Tensor")
    if position.ndim != 2 or position.shape[1] != 2:
        raise ValueError("position must have shape [N, 2]")
    if not position.is_floating_point():
        raise TypeError("position must use a floating-point dtype")
    if not bool(torch.all(torch.isfinite(position))):
        raise ValueError("position contains NaN or infinite values")
    if not isinstance(neighbors, int) or isinstance(neighbors, bool):
        raise TypeError("neighbors must be an integer")
    if neighbors < 1:
        raise ValueError("neighbors must be positive")
    event_count = position.shape[0]
    if event_count <= neighbors:
        raise ValueError(
            f"neighbors={neighbors} requires at least {neighbors + 1} events"
        )
    if target_chunk_size < 1 or source_chunk_size < 1:
        raise ValueError("bandwidth chunk sizes must be positive")
    minimum = float(minimum)
    if not math.isfinite(minimum) or minimum <= 0.0:
        raise ValueError("minimum bandwidth must be finite and positive")

    output = position.new_empty(event_count)
    infinity = torch.tensor(float("inf"), dtype=position.dtype, device=position.device)
    for target_begin in range(0, event_count, target_chunk_size):
        target_end = min(target_begin + target_chunk_size, event_count)
        target = position[target_begin:target_end]
        best_squared = torch.full(
            (target_end - target_begin, neighbors),
            float("inf"),
            dtype=position.dtype,
            device=position.device,
        )
        target_indices = torch.arange(
            target_begin, target_end, device=position.device
        )[:, None]
        for source_begin in range(0, event_count, source_chunk_size):
            source_end = min(source_begin + source_chunk_size, event_count)
            source = position[source_begin:source_end]
            displacement = target[:, None, :] - source[None, :, :]
            squared = displacement.square().sum(dim=-1)
            source_indices = torch.arange(
                source_begin, source_end, device=position.device
            )[None, :]
            squared = torch.where(
                target_indices == source_indices, infinity, squared
            )
            candidates = torch.cat((best_squared, squared), dim=1)
            best_squared = torch.topk(
                candidates, k=neighbors, dim=1, largest=False, sorted=False
            ).values
        kth_distance = torch.sqrt(best_squared.amax(dim=1))
        output[target_begin:target_end] = kth_distance.clamp_min(minimum)
    return output
