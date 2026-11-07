"""Readers for legacy finite-source patch files."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Mapping

import torch

from .geometry import FiniteSourceGeometry


def load_legacy_finite_source_geometry(
    path: str | Path,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
    distance_unit: str = "degree",
    coordinate_system: str = "EPSG:4326",
    depth_unit: str = "km",
    depth_positive: str = "down",
    metadata: Mapping[str, Any] | None = None,
) -> FiniteSourceGeometry:
    """Load a headerless legacy ``sub*.etas`` finite-source patch table.

    Each non-comment row must contain exactly five whitespace-separated
    fields::

        patch_id  longitude_or_x  latitude_or_y  depth  finite_source_id

    Rows are stably grouped by ``finite_source_id``. Their order inside each
    source group is preserved, which keeps legacy patch numbering and tau
    output alignment reproducible.
    """

    if dtype not in (torch.float32, torch.float64):
        raise TypeError("dtype must be torch.float32 or torch.float64")
    if depth_positive not in {"up", "down", "unspecified"}:
        raise ValueError("depth_positive must be 'up', 'down', or 'unspecified'")
    source_path = Path(path).expanduser()
    if not source_path.is_file():
        raise FileNotFoundError(
            f"finite-source geometry does not exist: {source_path}"
        )

    patch_id: list[int] = []
    finite_source_id: list[int] = []
    x: list[float] = []
    y: list[float] = []
    depth: list[float] = []
    with source_path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if len(fields) != 5:
                raise ValueError(
                    f"finite-source row {line_number} has {len(fields)} fields; "
                    "expected 5"
                )
            patch_id.append(
                _parse_integer(fields[0], "patch_id", line_number)
            )
            x.append(_parse_float(fields[1], "x", line_number))
            y.append(_parse_float(fields[2], "y", line_number))
            depth.append(_parse_float(fields[3], "depth", line_number))
            finite_source_id.append(
                _parse_integer(
                    fields[4], "finite_source_id", line_number
                )
            )
    if not patch_id:
        raise ValueError(f"finite-source geometry contains no patches: {source_path}")

    order = sorted(
        range(len(patch_id)), key=finite_source_id.__getitem__
    )
    was_grouped = order == list(range(len(order)))

    def reordered(values: list[Any]) -> list[Any]:
        return [values[index] for index in order]

    geometry_metadata = dict(metadata or {})
    x_label = "longitude" if distance_unit.lower() in {
        "degree",
        "degrees",
        "deg",
    } else "x"
    y_label = "latitude" if x_label == "longitude" else "y"
    geometry_metadata.update(
        {
            "source_path": str(source_path.resolve()),
            "source_sha256": _file_sha256(source_path),
            "source_format": source_path.suffix.lower().lstrip("."),
            "source_patch_count": len(patch_id),
            "source_has_header": False,
            "stable_source_group_sort_applied": not was_grouped,
            "columns": {
                "patch_id": "patch_id",
                "x": x_label,
                "y": y_label,
                "depth": "depth",
                "finite_source_id": "finite_source_id",
            },
            "distance_unit": distance_unit,
            "coordinate_kind": (
                "geographic" if x_label == "longitude" else "unspecified"
            ),
            "coordinate_system": coordinate_system,
            "depth_unit": depth_unit,
            "depth_positive": depth_positive,
        }
    )
    geometry = FiniteSourceGeometry(
        patch_id=torch.tensor(reordered(patch_id), dtype=torch.int64),
        finite_source_id=torch.tensor(
            reordered(finite_source_id), dtype=torch.int64
        ),
        position=torch.tensor(
            [[x[index], y[index]] for index in order], dtype=dtype
        ),
        depth=torch.tensor(reordered(depth), dtype=dtype),
        metadata=geometry_metadata,
    )
    return geometry.to(device=device, dtype=dtype)


def _parse_integer(value: str, column: str, line_number: int) -> int:
    try:
        converted = int(value)
    except ValueError as error:
        raise ValueError(
            f"column '{column}' contains a non-integer value at input "
            f"row {line_number}: {value!r}"
        ) from error
    return converted


def _parse_float(value: str, column: str, line_number: int) -> float:
    try:
        converted = float(value)
    except ValueError as error:
        raise ValueError(
            f"column '{column}' contains a non-numeric value at input "
            f"row {line_number}: {value!r}"
        ) from error
    if not math.isfinite(converted):
        raise ValueError(
            f"column '{column}' contains a non-finite value at input "
            f"row {line_number}"
        )
    return converted


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
