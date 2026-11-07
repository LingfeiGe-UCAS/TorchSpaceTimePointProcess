import pytest
import torch

from torchstpp.data import EventCatalog


def make_catalog() -> EventCatalog:
    return EventCatalog(
        event_id=torch.tensor([10, 11, 12, 13], dtype=torch.int64),
        time=torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64),
        position=torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
            dtype=torch.float64,
        ),
        magnitude=torch.tensor([3.0, 3.1, 3.2, 3.3], dtype=torch.float64),
        completeness_magnitude=torch.tensor(3.0, dtype=torch.float64),
        depth=torch.tensor([4.0, 5.0, 6.0, 7.0], dtype=torch.float64),
        inside=torch.tensor([True, False, True, False]),
    )


def test_take_contiguous_slice_reuses_catalog_contract():
    selected = make_catalog().take(slice(1, 3))

    assert selected.event_id.tolist() == [11, 12]
    assert selected.depth.tolist() == [5.0, 6.0]
    assert selected.inside.tolist() == [False, True]
    assert selected.sequence_index.tolist() == [1, 2]


def test_take_boolean_mask_preserves_time_order():
    selected = make_catalog().take(torch.tensor([True, False, True, False]))

    assert selected.event_id.tolist() == [10, 12]
    assert selected.time.tolist() == [0.0, 2.0]
    assert selected.sequence_index.tolist() == [0, 2]


def test_take_integer_indices_delegates_to_index_select():
    selected = make_catalog().take([0, 3])

    assert selected.event_id.tolist() == [10, 13]


def test_take_rejects_wrong_or_empty_boolean_mask():
    catalog = make_catalog()

    with pytest.raises(ValueError, match="shape"):
        catalog.take(torch.tensor([True, False]))
    with pytest.raises(ValueError, match="zero events"):
        catalog.take(torch.zeros(len(catalog), dtype=torch.bool))
