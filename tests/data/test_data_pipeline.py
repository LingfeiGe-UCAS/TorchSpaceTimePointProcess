import torch

from torchstpp.data import (
    EventCatalog,
    ObservationDomain,
    ProjectTransformer,
    resolve_domain,
)


def make_geographic_catalog() -> EventCatalog:
    return EventCatalog(
        event_id=torch.tensor([100, 101, 102, 103, 104], dtype=torch.int64),
        time=torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0], dtype=torch.float64),
        position=torch.tensor(
            [
                [101.0, 36.0],
                [101.1, 36.1],
                [101.2, 36.2],
                [101.3, 36.3],
                [101.4, 36.4],
            ],
            dtype=torch.float64,
        ),
        magnitude=torch.tensor([2.9, 3.0, 3.2, 3.4, 3.1], dtype=torch.float64),
        completeness_magnitude=torch.tensor(3.0, dtype=torch.float64),
        depth=torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64),
        metadata={
            "time_unit": "day",
            "coordinate_kind": "geographic",
            "coordinate_system": "EPSG:4326",
            "distance_unit": "degree",
        },
    )


def test_analysis_filter_preserves_original_event_ids():
    filtered = make_geographic_catalog().filter_for_analysis(
        time_zero=1.0, time_end=3.0
    )

    assert filtered.event_id.tolist() == [101, 102, 103]
    assert torch.all(filtered.magnitude >= filtered.completeness_magnitude)
    assert filtered.metadata["analysis_time_zero"] == 1.0


def test_local_transformer_round_trip_and_domain_masks():
    filtered = make_geographic_catalog().filter_for_analysis(
        time_zero=1.0, time_end=4.0
    )
    transformer = ProjectTransformer.fit_local(filtered)
    projected = filtered.project(transformer)
    restored = projected.inverse_project(transformer)

    assert torch.allclose(restored.position, filtered.position, atol=1e-12)
    domain = resolve_domain(
        projected,
        time_zero=1.0,
        time_start=2.0,
        time_end=4.0,
        padding=1.0,
    )
    assert domain.source_mask(projected).tolist() == [True, True, True, True]
    assert domain.target_mask(projected).tolist() == [False, True, True, True]


def test_domain_border_uses_same_forward_and_inverse_transformer():
    dtype = torch.float64
    domain = ObservationDomain(
        time_zero=torch.tensor(0.0, dtype=dtype),
        time_start=torch.tensor(1.0, dtype=dtype),
        time_end=torch.tensor(2.0, dtype=dtype),
        border=torch.tensor(
            [[101.0, 36.0], [101.2, 36.0], [101.2, 36.2], [101.0, 36.2]],
            dtype=dtype,
        ),
        metadata={
            "coordinate_kind": "geographic",
            "coordinate_system": "EPSG:4326",
            "distance_unit": "degree",
        },
    )
    transformer = ProjectTransformer.local_equirectangular(
        origin_longitude=101.1,
        origin_latitude=36.1,
    )

    restored = domain.project(transformer).inverse_project(transformer)

    assert torch.allclose(restored.border, domain.border, atol=1e-12)


def test_arbitrary_crs_transform_is_reversible():
    transformer = ProjectTransformer.from_crs(
        "EPSG:4326", "EPSG:32647", target_distance_unit="m"
    )
    points = torch.tensor([[101.5, 36.2], [101.6, 36.3]], dtype=torch.float64)

    restored = transformer.inverse(transformer.forward(points))

    assert torch.allclose(restored, points, atol=1e-8)
