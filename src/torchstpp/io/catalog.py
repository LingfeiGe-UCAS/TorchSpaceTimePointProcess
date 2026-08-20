"""File adapters for :class:`torchstpp.data.EventCatalog`."""

from __future__ import annotations

import csv
import hashlib
import math
import numbers
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from ..data import EventCatalog


@dataclass(frozen=True)
class CatalogColumns:
    """Map canonical catalog fields to columns in an input table.

    ``event_id``, ``depth``, and ``inside`` are optional.  If the event-ID
    column is absent, zero-based input row numbers are generated before the
    stable time sort.  Missing depth and region-mask columns become ``None``.
    """

    time: str = "time"
    x: str = "x"
    y: str = "y"
    magnitude: str = "magnitude"
    event_id: str | None = "event_id"
    depth: str | None = "depth"
    inside: str | None = "inside"

    @classmethod
    def geographic_earthquake(
        cls, *, event_id: str | None = None
    ) -> "CatalogColumns":
        """Columns used by the raw earthquake catalogs in the old project.

        Coordinates remain longitude/latitude.  This preset only maps fields;
        it deliberately does not project them.  Pass ``event_id="idx"`` or
        ``event_id="index"`` when that source column should be preserved.
        """

        return cls(
            x="longitude",
            y="latitude",
            magnitude="mag",
            event_id=event_id,
        )

    @classmethod
    def legacy_etas(cls) -> "CatalogColumns":
        """Columns in an already projected table from the earlier ETAS code."""

        return cls(x="x_proj", y="y_proj", magnitude="mag")

    @classmethod
    def from_mapping(
        cls, mapping: Mapping[str, str | None] | None
    ) -> "CatalogColumns":
        if mapping is None:
            return cls()
        valid = set(asdict(cls()))
        unknown = set(mapping) - valid
        if unknown:
            raise ValueError(f"unknown canonical column names: {sorted(unknown)}")
        return replace(cls(), **dict(mapping))


def load_catalog(
    path: str | Path,
    *,
    completeness_magnitude: float,
    columns: CatalogColumns | Mapping[str, str | None] | None = None,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
    sort_by_time: bool = True,
    delimiter: str | None = None,
    time_unit: str = "unspecified",
    distance_unit: str = "unspecified",
    coordinate_system: str = "unspecified",
    time_origin: str | float | None = None,
    time_offset: float = 0.0,
    time_zero: float | torch.Tensor | None = None,
    time_end: float | torch.Tensor | None = None,
    apply_analysis_filter: bool = True,
    time_duplicate_policy: str = "jitter",
    position_duplicate_policy: str = "jitter",
    position_minimum_separation_m: float = 0.05,
    depth_duplicate_policy: str = "jitter",
    depth_minimum_separation: float = 0.05,
    warn_on_duplicate_resolution: bool = True,
    depth_unit: str = "unspecified",
    depth_positive: str = "unspecified",
    metadata: Mapping[str, Any] | None = None,
) -> EventCatalog:
    """Read a CSV/TSV/ETAS/Parquet event table and return validated tensors.

    Loading, stable sorting, and configured duplicate resolution happen on the
    CPU. By default finite-precision time ties and exactly repeated positions
    receive deterministic minimal perturbations and emit ``UserWarning``.
    Select ``keep`` or ``error`` explicitly to change that policy. The result
    is moved to ``device`` only afterward, keeping file I/O independent of CUDA.
    By default the returned catalog is explicitly restricted to
    ``magnitude >= completeness_magnitude`` and ``time_zero <= time <=
    time_end``.  Set ``apply_analysis_filter=False`` only when a raw canonical
    catalog is intentionally required.  Projection remains a separate step.

    Parameters
    ----------
    path:
        Input file ending in ``.csv``, ``.tsv``, ``.etas``, ``.parquet``, or
        ``.pq``.  ``.etas`` files are read as whitespace-separated tables.
    completeness_magnitude:
        The catalog threshold used by ETAS productivity and spatial scaling.
        It is recorded but is not applied as a hidden row filter.
    columns:
        Canonical-to-input column mapping.  Either ``CatalogColumns`` or a
        partial mapping such as ``{"magnitude": "mag", "x": "x_proj"}``.
    dtype:
        Floating tensor type; currently ``torch.float32`` or ``torch.float64``.
    device:
        Final tensor device.  An unavailable CUDA device raises an error.
    sort_by_time:
        Apply a stable ascending time sort.  Equal-time rows retain input order.
    time_offset:
        Numeric offset subtracted from every source time before casting to the
        requested tensor dtype.  Use this to preserve float32 resolution when
        a file stores large absolute day/second values.
    time_zero, time_end:
        Source-history bounds expressed after ``time_offset`` is subtracted.
        Omitted bounds use the minimum and maximum loaded catalog times.
    """

    source_path = Path(path).expanduser()
    if not source_path.is_file():
        raise FileNotFoundError(f"event catalog does not exist: {source_path}")
    source_sha256 = _file_sha256(source_path)
    if isinstance(columns, CatalogColumns):
        column_spec = columns
        explicitly_required_optional = {
            canonical
            for canonical, default_name in (
                ("event_id", "event_id"),
                ("depth", "depth"),
                ("inside", "inside"),
            )
            if getattr(columns, canonical) not in {None, default_name}
        }
    else:
        explicitly_required_optional = {
            canonical
            for canonical in ("event_id", "depth", "inside")
            if columns is not None
            and canonical in columns
            and columns[canonical] is not None
        }
        column_spec = CatalogColumns.from_mapping(columns)
    suffix = source_path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        actual_delimiter = delimiter or ("\t" if suffix == ".tsv" else ",")
        table = _read_delimited(source_path, actual_delimiter)
    elif suffix == ".etas":
        if delimiter is not None:
            raise ValueError("delimiter is only valid for CSV or TSV input")
        table = _read_whitespace(source_path)
    elif suffix in {".parquet", ".pq"}:
        if delimiter is not None:
            raise ValueError("delimiter is only valid for CSV or TSV input")
        table = _read_parquet(source_path)
    else:
        raise ValueError(
            "unsupported catalog format; expected .csv, .tsv, .etas, .parquet, or .pq"
        )

    cpu_catalog = _table_to_catalog(
        table,
        source_path=source_path,
        source_sha256=source_sha256,
        completeness_magnitude=completeness_magnitude,
        columns=column_spec,
        explicitly_required_optional=explicitly_required_optional,
        dtype=dtype,
        sort_by_time=sort_by_time,
        time_unit=time_unit,
        distance_unit=distance_unit,
        coordinate_system=coordinate_system,
        time_origin=time_origin,
        time_offset=time_offset,
        depth_unit=depth_unit,
        depth_positive=depth_positive,
        metadata=metadata,
    )
    cpu_catalog = cpu_catalog.resolve_duplicates(
        time_policy=time_duplicate_policy,
        position_policy=position_duplicate_policy,
        position_minimum_separation_m=position_minimum_separation_m,
        depth_policy=depth_duplicate_policy,
        depth_minimum_separation=depth_minimum_separation,
        warn=warn_on_duplicate_resolution,
    )
    if not isinstance(apply_analysis_filter, bool):
        raise TypeError("apply_analysis_filter must be a boolean")
    if apply_analysis_filter:
        cpu_catalog = cpu_catalog.filter_for_analysis(
            time_zero=time_zero,
            time_end=time_end,
        )
    elif time_zero is not None or time_end is not None:
        raise ValueError(
            "time_zero/time_end require apply_analysis_filter=True"
        )
    return cpu_catalog.to(device=device, dtype=dtype)


def _read_delimited(path: Path, delimiter: str) -> dict[str, list[Any]]:
    if len(delimiter) != 1:
        raise ValueError("delimiter must be exactly one character")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"catalog has no header row: {path}")
        if len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("catalog header contains duplicate column names")
        table = {name: [] for name in reader.fieldnames}
        for line_number, row in enumerate(reader, start=2):
            if None in row or any(row[name] is None for name in reader.fieldnames):
                raise ValueError(
                    f"catalog row {line_number} has a different number of fields "
                    "than the header"
                )
            for name in reader.fieldnames:
                table[name].append(row[name])
    return table


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_whitespace(path: Path) -> dict[str, list[Any]]:
    header: list[str] | None = None
    table: dict[str, list[Any]] = {}
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if header is None:
                header = fields
                if len(set(header)) != len(header):
                    raise ValueError("catalog header contains duplicate column names")
                table = {name: [] for name in header}
                continue
            if len(fields) != len(header):
                raise ValueError(
                    f"catalog row {line_number} has {len(fields)} fields; "
                    f"expected {len(header)}"
                )
            for name, value in zip(header, fields):
                table[name].append(value)
    if header is None:
        raise ValueError(f"catalog has no header row: {path}")
    return table


def _read_parquet(path: Path) -> dict[str, list[Any]]:
    try:
        import polars as pl  # type: ignore[import-not-found]
    except ImportError:
        try:
            import pyarrow.parquet as pq  # type: ignore[import-not-found]
        except ImportError as error:
            raise ImportError(
                "Parquet input requires Polars or PyArrow; install "
                "the package with `pip install -e '.[parquet]'`"
            ) from error
        return {
            name: list(values)
            for name, values in pq.read_table(path).to_pydict().items()
        }

    frame = pl.read_parquet(path)
    return {name: list(values) for name, values in frame.to_dict(as_series=False).items()}


def _table_to_catalog(
    table: Mapping[str, Sequence[Any]],
    *,
    source_path: Path,
    source_sha256: str,
    completeness_magnitude: float,
    columns: CatalogColumns,
    explicitly_required_optional: set[str],
    dtype: torch.dtype,
    sort_by_time: bool,
    time_unit: str,
    distance_unit: str,
    coordinate_system: str,
    time_origin: str | float | None,
    time_offset: float,
    depth_unit: str,
    depth_positive: str,
    metadata: Mapping[str, Any] | None,
) -> EventCatalog:
    if dtype not in (torch.float32, torch.float64):
        raise TypeError("catalog dtype must be torch.float32 or torch.float64")
    if depth_positive not in {"up", "down", "unspecified"}:
        raise ValueError("depth_positive must be 'up', 'down', or 'unspecified'")
    if columns.event_id is None and "event_id" in explicitly_required_optional:
        raise ValueError("event_id cannot be required when its column name is None")
    required = {
        "time": columns.time,
        "x": columns.x,
        "y": columns.y,
        "magnitude": columns.magnitude,
    }
    missing = [
        f"{canonical} ({name})"
        for canonical, name in required.items()
        if name not in table
    ]
    if missing:
        raise ValueError(f"catalog is missing required columns: {', '.join(missing)}")
    missing_optional = [
        f"{canonical} ({getattr(columns, canonical)})"
        for canonical in sorted(explicitly_required_optional)
        if getattr(columns, canonical) not in table
    ]
    if missing_optional:
        raise ValueError(
            "catalog is missing explicitly requested columns: "
            + ", ".join(missing_optional)
        )

    lengths = {name: len(values) for name, values in table.items()}
    if not lengths:
        raise ValueError("catalog has no columns")
    if len(set(lengths.values())) != 1:
        raise ValueError(f"catalog columns have inconsistent lengths: {lengths}")
    event_count = next(iter(lengths.values()))
    if event_count < 1:
        raise ValueError("catalog contains no events")

    try:
        normalized_time_offset = float(time_offset)
    except (TypeError, ValueError) as error:
        raise TypeError("time_offset must be a real number") from error
    if not math.isfinite(normalized_time_offset):
        raise ValueError("time_offset must be finite")

    source_time = _float_values(table[columns.time], columns.time)
    time = [value - normalized_time_offset for value in source_time]
    x = _float_values(table[columns.x], columns.x)
    y = _float_values(table[columns.y], columns.y)
    magnitude = _float_values(table[columns.magnitude], columns.magnitude)
    if columns.event_id is not None and columns.event_id in table:
        event_id = _integer_values(table[columns.event_id], columns.event_id)
        generated_event_id = False
    else:
        event_id = list(range(event_count))
        generated_event_id = True
    depth = (
        _float_values(table[columns.depth], columns.depth)
        if columns.depth is not None and columns.depth in table
        else None
    )
    inside = (
        _boolean_values(table[columns.inside], columns.inside)
        if columns.inside is not None and columns.inside in table
        else None
    )

    if len(set(event_id)) != event_count:
        raise ValueError("event_id values must be unique")
    try:
        normalized_completeness_magnitude = float(completeness_magnitude)
    except (TypeError, ValueError) as error:
        raise TypeError("completeness_magnitude must be a real number") from error
    if not math.isfinite(normalized_completeness_magnitude):
        raise ValueError("completeness_magnitude must be finite")

    order = list(range(event_count))
    was_sorted = all(time[index] >= time[index - 1] for index in range(1, event_count))
    if sort_by_time:
        order.sort(key=time.__getitem__)
    elif not was_sorted:
        raise ValueError("time is not sorted; pass sort_by_time=True to sort stably")

    def reordered(values: Sequence[Any]) -> list[Any]:
        return [values[index] for index in order]

    catalog_metadata = dict(metadata or {})
    x_name = columns.x.lower()
    y_name = columns.y.lower()
    if x_name in {"longitude", "lon"} and y_name in {"latitude", "lat"}:
        coordinate_kind = "geographic"
    elif "proj" in x_name or "proj" in y_name or distance_unit.lower() in {
        "m",
        "meter",
        "meters",
        "km",
        "kilometer",
        "kilometers",
    }:
        coordinate_kind = "projected"
    else:
        coordinate_kind = "unspecified"
    catalog_metadata.update(
        {
            "source_path": str(source_path.resolve()),
            "source_sha256": source_sha256,
            "source_format": source_path.suffix.lower().lstrip("."),
            "source_event_count": event_count,
            "generated_event_id": generated_event_id,
            "stable_time_sort_applied": bool(sort_by_time and not was_sorted),
            "time_unit": time_unit,
            "time_origin": time_origin,
            "source_time_offset": normalized_time_offset,
            "distance_unit": distance_unit,
            "coordinate_kind": coordinate_kind,
            "coordinate_system": coordinate_system,
            "depth_unit": depth_unit,
            "depth_positive": depth_positive,
            "columns": asdict(columns),
        }
    )

    return EventCatalog(
        event_id=torch.tensor(reordered(event_id), dtype=torch.int64),
        time=torch.tensor(reordered(time), dtype=dtype),
        position=torch.tensor(
            [[x[index], y[index]] for index in order], dtype=dtype
        ),
        magnitude=torch.tensor(reordered(magnitude), dtype=dtype),
        completeness_magnitude=torch.tensor(
            normalized_completeness_magnitude, dtype=dtype
        ),
        depth=None if depth is None else torch.tensor(reordered(depth), dtype=dtype),
        inside=None
        if inside is None
        else torch.tensor(reordered(inside), dtype=torch.bool),
        metadata=catalog_metadata,
    )


def _float_values(values: Sequence[Any], column: str) -> list[float]:
    result: list[float] = []
    for row_number, value in enumerate(values, start=2):
        try:
            converted = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"column '{column}' contains a non-numeric value at input "
                f"row {row_number}: {value!r}"
            ) from error
        if not math.isfinite(converted):
            raise ValueError(
                f"column '{column}' contains a non-finite value at input row {row_number}"
            )
        result.append(converted)
    return result


def _integer_values(values: Sequence[Any], column: str) -> list[int]:
    result: list[int] = []
    for row_number, value in enumerate(values, start=2):
        if isinstance(value, bool):
            valid = False
        elif isinstance(value, numbers.Integral):
            valid = True
            converted = int(value)
        elif isinstance(value, numbers.Real):
            numeric_value = float(value)
            valid = math.isfinite(numeric_value) and numeric_value.is_integer()
            converted = int(numeric_value) if valid else 0
        else:
            try:
                text = str(value).strip()
                converted = int(text)
                valid = text not in {"", "+", "-"}
            except (TypeError, ValueError):
                valid = False
                converted = 0
        if not valid:
            raise ValueError(
                f"column '{column}' contains a non-integer ID at input row {row_number}: {value!r}"
            )
        result.append(converted)
    return result


def _boolean_values(values: Sequence[Any], column: str) -> list[bool]:
    truthy = {True, 1, "1", "true", "t", "yes", "y"}
    falsy = {False, 0, "0", "false", "f", "no", "n"}
    result: list[bool] = []
    for row_number, value in enumerate(values, start=2):
        normalized = value.strip().lower() if isinstance(value, str) else value
        if normalized in truthy:
            result.append(True)
        elif normalized in falsy:
            result.append(False)
        else:
            raise ValueError(
                f"column '{column}' contains an invalid boolean at input "
                f"row {row_number}: {value!r}"
            )
    return result
