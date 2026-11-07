from pathlib import Path

import pytest
import torch

from torchstpp import (
    CatalogColumns,
    FiniteSourceGeometry,
    ProjectTransformer,
    load_catalog,
    load_legacy_finite_source_geometry,
)
from torchstpp.data import EventCatalog


def make_catalog() -> EventCatalog:
    return EventCatalog(
        event_id=torch.tensor([10, 11, 12, 13], dtype=torch.int64),
        time=torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64),
        position=torch.tensor(
            [[136.0, 37.0], [136.1, 37.1], [136.2, 37.2], [136.3, 37.3]],
            dtype=torch.float64,
        ),
        magnitude=torch.tensor([2.0, 2.1, 2.2, 2.3], dtype=torch.float64),
        completeness_magnitude=torch.tensor(2.0, dtype=torch.float64),
        depth=torch.tensor([4.0, 5.0, 6.0, 7.0], dtype=torch.float64),
        finite_source_id=torch.tensor([0, 2, 0, 1], dtype=torch.int64),
        metadata={
            "coordinate_kind": "geographic",
            "coordinate_system": "EPSG:4326",
            "distance_unit": "degree",
        },
    )


def test_finite_source_id_is_preserved_by_catalog_operations():
    catalog = make_catalog()

    selected = catalog.take([1, 3])
    filtered = catalog.filter_for_analysis(time_zero=1.0, time_end=3.0)
    converted = catalog.to(dtype=torch.float32)
    transformer = ProjectTransformer.legacy_local_approx(
        reference_latitude=37.15
    )
    restored = catalog.project(transformer).inverse_project(transformer)

    assert selected.finite_source_id.tolist() == [2, 1]
    assert filtered.finite_source_id.tolist() == [2, 0, 1]
    assert converted.finite_source_id.dtype == torch.int64
    assert torch.equal(restored.finite_source_id, catalog.finite_source_id)
    assert catalog.summary()["finite_source_events"] == 2
    assert catalog.summary()["finite_source_groups"] == 2


def test_finite_source_id_validation_rejects_invalid_values():
    values = make_catalog()

    with pytest.raises(TypeError, match="dtype torch.int64"):
        EventCatalog(
            event_id=values.event_id,
            time=values.time,
            position=values.position,
            magnitude=values.magnitude,
            completeness_magnitude=values.completeness_magnitude,
            finite_source_id=values.finite_source_id.to(torch.int32),
        )
    with pytest.raises(ValueError, match="nonnegative"):
        EventCatalog(
            event_id=values.event_id,
            time=values.time,
            position=values.position,
            magnitude=values.magnitude,
            completeness_magnitude=values.completeness_magnitude,
            finite_source_id=torch.tensor([0, 1, -1, 0]),
        )


def test_load_headerless_catalog_with_explicit_column_names(tmp_path: Path):
    path = tmp_path / "legacy.etas"
    path.write_text(
        "10 136.2 37.2 2.1 2.0 5.0 0 1\n"
        "11 136.1 37.1 6.5 1.0 7.0 3 1\n",
        encoding="utf-8",
    )
    columns = CatalogColumns(
        event_id="idx",
        x="longitude",
        y="latitude",
        magnitude="mag",
        time="time",
        depth="depth",
        finite_source_id="type",
        inside="unknown",
    )

    catalog = load_catalog(
        path,
        completeness_magnitude=2.0,
        columns=columns,
        column_names=(
            "idx",
            "longitude",
            "latitude",
            "mag",
            "time",
            "depth",
            "type",
            "unknown",
        ),
        apply_analysis_filter=False,
        time_duplicate_policy="keep",
        position_duplicate_policy="keep",
        depth_duplicate_policy="keep",
        warn_on_duplicate_resolution=False,
        distance_unit="degree",
        coordinate_system="EPSG:4326",
    )

    assert catalog.event_id.tolist() == [11, 10]
    assert catalog.finite_source_id.tolist() == [3, 0]
    assert catalog.inside.tolist() == [True, True]
    assert catalog.metadata["stable_time_sort_applied"] is True


def test_legacy_geometry_loader_groups_stably_and_preserves_duplicates(
    tmp_path: Path,
):
    path = tmp_path / "subNoto.etas"
    path.write_text(
        "10 137.2 37.5 3.0 2\n"
        "11 136.5 37.0 0.5 1\n"
        "12 137.2 37.5 5.0 2\n"
        "13 136.5 37.0 2.5 1\n",
        encoding="utf-8",
    )

    geometry = load_legacy_finite_source_geometry(path)

    assert geometry.patch_id.tolist() == [11, 13, 10, 12]
    assert geometry.finite_source_id.tolist() == [1, 1, 2, 2]
    assert geometry.group_ids.tolist() == [1, 2]
    assert geometry.group_offsets.tolist() == [0, 2, 4]
    assert geometry.group_counts.tolist() == [2, 2]
    assert geometry.group_slice(2) == slice(2, 4)
    assert torch.equal(geometry.position[0], geometry.position[1])
    assert geometry.metadata["stable_source_group_sort_applied"] is True

    converted = geometry.to(dtype=torch.float32)
    shifted = geometry.with_position(
        geometry.position + 1.0, coordinate_system="test"
    )
    assert converted.dtype == torch.float32
    assert converted.patch_id.dtype == torch.int64
    assert shifted.metadata["coordinate_system"] == "test"
    assert torch.equal(shifted.group_offsets, geometry.group_offsets)


def test_geometry_validation_requires_unique_grouped_positive_ids():
    kwargs = {
        "patch_id": torch.tensor([1, 2], dtype=torch.int64),
        "finite_source_id": torch.tensor([1, 2], dtype=torch.int64),
        "position": torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
        "depth": torch.tensor([0.5, 2.5]),
    }

    with pytest.raises(ValueError, match="unique"):
        FiniteSourceGeometry(
            **{**kwargs, "patch_id": torch.tensor([1, 1])}
        )
    with pytest.raises(ValueError, match="positive"):
        FiniteSourceGeometry(
            **{**kwargs, "finite_source_id": torch.tensor([0, 1])}
        )
    with pytest.raises(ValueError, match="grouped"):
        FiniteSourceGeometry(
            patch_id=torch.tensor([1, 2, 3]),
            finite_source_id=torch.tensor([1, 2, 1]),
            position=torch.zeros((3, 2)),
            depth=torch.ones(3),
        )
