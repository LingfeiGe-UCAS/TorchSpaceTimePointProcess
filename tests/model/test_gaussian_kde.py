from types import SimpleNamespace

import torch

from torchstpp.models.backgrounds import GaussianKDEBackground, KDEState


def make_state():
    return KDEState(
        position=torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]),
        bandwidth=torch.tensor([0.2, 0.3, 0.4]),
        weights=torch.tensor([0.5, 0.3, 0.2]),
        duration=10.0,
    )


def test_kde_evaluate_matches_legacy_formula_and_chunking():
    state = make_state()
    target = SimpleNamespace(
        position=torch.tensor([[0.0, 0.0], [0.5, 0.5], [2.0, 1.0]])
    )
    background = GaussianKDEBackground(target_chunk_size=2, source_chunk_size=2)
    actual = background.evaluate(target, state)

    displacement = target.position[:, None, :] - state.position[None, :, :]
    distance_squared = displacement.square().sum(dim=-1)
    bandwidth_squared = state.bandwidth.square()
    legacy_kernel = torch.exp(
        -distance_squared / (2.0 * bandwidth_squared)
    ) / (2.0 * torch.pi * bandwidth_squared)
    expected = (legacy_kernel * state.weights).sum(dim=1) / state.duration
    assert torch.allclose(actual, expected)


def test_kde_integral_matches_legacy_background_term():
    state = make_state()
    domain = SimpleNamespace(
        time_start=0.0,
        time_end=10.0,
        border=torch.tensor([
            [-100.0, -100.0], [100.0, -100.0],
            [100.0, 100.0], [-100.0, 100.0],
        ]),
    )
    background = GaussianKDEBackground(source_chunk_size=2)
    actual = background.integrate(state, domain)
    # On an effectively infinite polygon, every Gaussian mass is one. With
    # observation duration == KDE duration, this is the old sum(phi_i).
    assert torch.allclose(actual, state.weights.sum(), atol=1.0e-5)


def test_kde_angular_integral_matches_legacy_pgauss():
    state = make_state()
    radius = torch.tensor([
        [0.0, 0.1, 1.0],
        [0.2, 0.5, 2.0],
        [0.3, 0.8, 3.0],
    ])
    actual = GaussianKDEBackground.spatial_angular_integral(radius, state)
    expected = (
        1.0
        - torch.exp(
            -radius.square() / (2.0 * state.bandwidth[:, None].square())
        )
    ) / (2.0 * torch.pi)
    assert torch.allclose(actual, expected)


def test_kde_state_validation():
    state = make_state()
    state.bandwidth[0] = 0.0
    try:
        state.validate()
    except ValueError:
        return
    raise AssertionError("zero bandwidth must be rejected")
