import math

import torch

from torchstpp.data import ProjectTransformer


def test_legacy_local_approx_matches_old_formula_and_round_trips():
    points = torch.tensor([[100.0, 30.0], [101.5, 31.0]], dtype=torch.float64)
    transformer = ProjectTransformer.legacy_local_approx(reference_latitude=30.5)

    projected = transformer.forward(points)
    expected_x = points[:, 0] * math.cos(math.radians(30.5))

    assert torch.allclose(projected[:, 0], expected_x)
    assert torch.equal(projected[:, 1], points[:, 1])
    assert torch.allclose(transformer.inverse(projected), points)
    assert transformer.target_distance_unit == "scaled_degree"
