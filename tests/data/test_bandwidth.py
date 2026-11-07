import torch

from torchstpp.data import knn_bandwidth


def test_knn_bandwidth_matches_dense_kth_neighbor_with_minimum():
    position = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [3.0, 0.0], [3.0, 4.0]],
        dtype=torch.float64,
    )
    actual = knn_bandwidth(
        position,
        neighbors=2,
        minimum=2.5,
        target_chunk_size=2,
        source_chunk_size=2,
    )
    distance = torch.cdist(position, position)
    distance.fill_diagonal_(torch.inf)
    expected = torch.topk(distance, 2, largest=False, dim=1).values[:, -1]
    expected = expected.clamp_min(2.5)
    assert torch.allclose(actual, expected)
