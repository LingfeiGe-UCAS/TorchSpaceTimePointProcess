"""Validated tensor representation of finite-source patch geometry."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Mapping

import torch


_SUPPORTED_FLOAT_DTYPES = (torch.float32, torch.float64)


@dataclass(frozen=True, repr=False)
class FiniteSourceGeometry:
    """Finite-source patches grouped by their parent source identifier.

    ``finite_source_id`` must be positive and non-decreasing. This compact
    grouped representation keeps all patches for one physical source
    contiguous without expanding them into event-catalog rows. ``group_ids``
    and ``group_offsets`` are derived during validation; group ``g`` occupies
    ``group_offsets[g]:group_offsets[g + 1]``.
    """

    patch_id: torch.Tensor
    finite_source_id: torch.Tensor
    position: torch.Tensor
    depth: torch.Tensor
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)
    group_ids: torch.Tensor = field(init=False, repr=False, compare=False)
    group_offsets: torch.Tensor = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        tensors = {
            "patch_id": self.patch_id,
            "finite_source_id": self.finite_source_id,
            "position": self.position,
            "depth": self.depth,
        }
        for name, value in tensors.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")

        patch_count = self.patch_id.shape[0] if self.patch_id.ndim == 1 else -1
        if patch_count < 1:
            raise ValueError("patch_id must be a non-empty one-dimensional tensor")
        if self.finite_source_id.shape != (patch_count,):
            raise ValueError(
                f"finite_source_id must have shape [{patch_count}]"
            )
        if self.position.shape != (patch_count, 2):
            raise ValueError(f"position must have shape [{patch_count}, 2]")
        if self.depth.shape != (patch_count,):
            raise ValueError(f"depth must have shape [{patch_count}]")

        if self.patch_id.dtype != torch.int64:
            raise TypeError("patch_id must have dtype torch.int64")
        if self.finite_source_id.dtype != torch.int64:
            raise TypeError("finite_source_id must have dtype torch.int64")
        if self.position.dtype not in _SUPPORTED_FLOAT_DTYPES:
            raise TypeError("position dtype must be torch.float32 or torch.float64")
        if self.depth.dtype != self.position.dtype:
            raise TypeError(
                "depth must have the same floating dtype as position"
            )

        for name, value in tensors.items():
            if value.device != self.position.device:
                raise ValueError(
                    f"{name} must be on the same device as position "
                    f"({self.position.device})"
                )
        if not bool(torch.all(torch.isfinite(self.position))):
            raise ValueError("position contains NaN or infinite values")
        if not bool(torch.all(torch.isfinite(self.depth))):
            raise ValueError("depth contains NaN or infinite values")
        if bool(torch.any(self.depth < 0)):
            raise ValueError("depth values must be nonnegative")
        if torch.unique(self.patch_id).numel() != patch_count:
            raise ValueError("patch_id values must be unique")
        if bool(torch.any(self.finite_source_id <= 0)):
            raise ValueError("finite_source_id values must be positive")
        if patch_count > 1 and not bool(
            torch.all(self.finite_source_id[1:] >= self.finite_source_id[:-1])
        ):
            raise ValueError(
                "finite_source_id must be grouped in non-decreasing order"
            )
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")

        starts_mask = torch.ones(
            patch_count, dtype=torch.bool, device=self.position.device
        )
        if patch_count > 1:
            starts_mask[1:] = (
                self.finite_source_id[1:] != self.finite_source_id[:-1]
            )
        starts = torch.nonzero(starts_mask, as_tuple=False).flatten()
        offsets = torch.cat(
            (
                starts,
                starts.new_tensor([patch_count], dtype=torch.int64),
            )
        )
        object.__setattr__(self, "group_ids", self.finite_source_id[starts])
        object.__setattr__(self, "group_offsets", offsets)
        object.__setattr__(self, "metadata", copy.deepcopy(dict(self.metadata)))

    def __len__(self) -> int:
        return self.patch_id.shape[0]

    def __repr__(self) -> str:
        return (
            f"FiniteSourceGeometry(patches={len(self)}, groups={self.group_count}, "
            f"device='{self.device}', dtype={self.dtype})"
        )

    @property
    def device(self) -> torch.device:
        return self.position.device

    @property
    def dtype(self) -> torch.dtype:
        return self.position.dtype

    @property
    def group_count(self) -> int:
        return self.group_ids.numel()

    @property
    def group_counts(self) -> torch.Tensor:
        return self.group_offsets[1:] - self.group_offsets[:-1]

    def group_slice(self, finite_source_id: int) -> slice:
        """Return the contiguous patch slice for ``finite_source_id``."""

        source_id = int(finite_source_id)
        matches = torch.nonzero(
            self.group_ids == source_id, as_tuple=False
        ).flatten()
        if matches.numel() == 0:
            raise KeyError(f"finite source ID {source_id} has no patches")
        group_index = int(matches[0].item())
        return slice(
            int(self.group_offsets[group_index].item()),
            int(self.group_offsets[group_index + 1].item()),
        )

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        *,
        non_blocking: bool = False,
    ) -> "FiniteSourceGeometry":
        """Return the geometry on ``device`` and in floating ``dtype``."""

        target_device = self.device if device is None else torch.device(device)
        target_dtype = self.dtype if dtype is None else dtype
        if target_dtype not in _SUPPORTED_FLOAT_DTYPES:
            raise TypeError("dtype must be torch.float32 or torch.float64")
        if target_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device '{target_device}' was requested, but CUDA is "
                "not available"
            )
        float_kwargs = {
            "device": target_device,
            "dtype": target_dtype,
            "non_blocking": non_blocking,
        }
        device_kwargs = {
            "device": target_device,
            "non_blocking": non_blocking,
        }
        metadata = copy.deepcopy(dict(self.metadata))
        metadata.update(
            {
                "tensor_dtype": str(target_dtype),
                "tensor_device": str(target_device),
            }
        )
        return FiniteSourceGeometry(
            patch_id=self.patch_id.to(**device_kwargs),
            finite_source_id=self.finite_source_id.to(**device_kwargs),
            position=self.position.to(**float_kwargs),
            depth=self.depth.to(**float_kwargs),
            metadata=metadata,
        )

    def cpu(self) -> "FiniteSourceGeometry":
        return self.to(device="cpu")

    def cuda(
        self, device: torch.device | str = "cuda:0"
    ) -> "FiniteSourceGeometry":
        return self.to(device=device)

    def with_position(
        self,
        position: torch.Tensor,
        **metadata_updates: Any,
    ) -> "FiniteSourceGeometry":
        """Return a copy with new positions and updated coordinate metadata."""

        metadata = copy.deepcopy(dict(self.metadata))
        metadata.update(metadata_updates)
        return FiniteSourceGeometry(
            patch_id=self.patch_id,
            finite_source_id=self.finite_source_id,
            position=position,
            depth=self.depth,
            metadata=metadata,
        )

    def summary(self) -> dict[str, Any]:
        """Return a compact JSON-serializable geometry summary."""

        return {
            "patches": len(self),
            "groups": self.group_count,
            "group_ids": self.group_ids.detach().cpu().tolist(),
            "group_counts": self.group_counts.detach().cpu().tolist(),
            "device": str(self.device),
            "dtype": str(self.dtype).removeprefix("torch."),
            "position_min": self.position.amin(dim=0).detach().cpu().tolist(),
            "position_max": self.position.amax(dim=0).detach().cpu().tolist(),
            "depth_range": [
                float(self.depth.amin().item()),
                float(self.depth.amax().item()),
            ],
        }
