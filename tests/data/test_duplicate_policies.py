import warnings

import torch

from torchstpp.data import EventCatalog


def _catalog() -> EventCatalog:
    dtype = torch.float32
    return EventCatalog(
        event_id=torch.arange(4, dtype=torch.int64),
        time=torch.tensor([10000.0, 10000.0, 10000.0, 10001.0], dtype=dtype),
        position=torch.tensor(
            [[140.0, 35.0], [140.0, 35.0], [140.0, 35.0], [141.0, 36.0]],
            dtype=dtype,
        ),
        magnitude=torch.full((4,), 4.5, dtype=dtype),
        completeness_magnitude=torch.tensor(4.5, dtype=dtype),
        depth=torch.tensor([8.0, 8.0, 8.0, 9.0], dtype=dtype),
        metadata={
            "coordinate_kind": "geographic",
            "distance_unit": "degree",
            "coordinate_system": "EPSG:4326",
            "depth_unit": "km",
        },
    )


def test_explicit_jitter_uses_representable_steps_and_unique_coordinates():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _catalog().resolve_duplicates(depth_policy="jitter")

    assert torch.all(result.time[1:] > result.time[:-1])
    assert result.time[1] == torch.nextafter(
        result.time[0], result.time.new_tensor(float("inf"))
    )
    assert torch.unique(result.position, dim=0).shape[0] == 4
    assert torch.unique(result.depth).shape[0] == 4
    audit = result.metadata["duplicate_resolution"]
    assert audit["time_modified_events"] == 2
    assert audit["position_modified_events"] == 2
    assert audit["depth_modified_events"] == 2
    assert len(caught) == 3


def test_keep_preserves_ties_and_error_rejects_them():
    catalog = _catalog()
    kept = catalog.resolve_duplicates(
        time_policy="keep", position_policy="keep", depth_policy="keep", warn=False
    )
    assert torch.equal(kept.time, catalog.time)
    assert torch.equal(kept.position, catalog.position)
    assert torch.equal(kept.depth, catalog.depth)

    try:
        catalog.resolve_duplicates(time_policy="error", position_policy="keep")
    except ValueError as error:
        assert "time ties" in str(error)
    else:
        raise AssertionError("time_policy='error' must reject tied times")

    try:
        catalog.resolve_duplicates(time_policy="keep", position_policy="error")
    except ValueError as error:
        assert "repeated positions" in str(error)
    else:
        raise AssertionError("position_policy='error' must reject duplicates")

    try:
        catalog.resolve_duplicates(
            time_policy="keep", position_policy="keep", depth_policy="error"
        )
    except ValueError as error:
        assert "repeated depths" in str(error)
    else:
        raise AssertionError("depth_policy='error' must reject duplicates")
