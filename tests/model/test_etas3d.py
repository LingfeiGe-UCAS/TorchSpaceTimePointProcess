from dataclasses import dataclass

import torch

from torchstpp.integration import InfiniteSpaceIntegrator
from torchstpp.likelihood import ExactLikelihood
from torchstpp.models import ParameterConfig
from torchstpp.models.backgrounds import GaussianBetaKDEBackground, KDEState
from torchstpp.models.components import conditional_beta_depth_density
from torchstpp.models.etas import ETAS2D, ETAS2DParameters, ETAS3D, ETAS3DParameters


@dataclass
class Catalog3DStub:
    time: torch.Tensor
    position: torch.Tensor
    depth: torch.Tensor
    magnitude: torch.Tensor
    completeness_magnitude: float

    def take(self, index):
        return Catalog3DStub(
            time=self.time[index],
            position=self.position[index],
            depth=self.depth[index],
            magnitude=self.magnitude[index],
            completeness_magnitude=self.completeness_magnitude,
        )


@dataclass
class DomainStub:
    time_zero: float
    time_start: float
    time_end: float
    border: torch.Tensor

    @property
    def area(self):
        return 4.0

    def contains(self, position):
        return (
            (position[:, 0] >= 0) & (position[:, 0] <= 2)
            & (position[:, 1] >= 0) & (position[:, 1] <= 2)
        )

    def target_mask(self, catalog):
        return (
            (catalog.time >= self.time_start)
            & (catalog.time <= self.time_end)
            & self.contains(catalog.position)
        )

    def source_mask(self, catalog):
        return (catalog.time >= self.time_zero) & (catalog.time <= self.time_end)


def parameters_3d():
    return ETAS3DParameters(
        mu=ParameterConfig(0.4),
        amplitude=ParameterConfig(0.2),
        alpha=ParameterConfig(0.8),
        c=ParameterConfig(0.1),
        p=ParameterConfig(1.2),
        d=ParameterConfig(0.3),
        q=ParameterConfig(1.6),
        gamma=ParameterConfig(0.5),
        eta=ParameterConfig(8.0),
        delta=ParameterConfig(0.2),
    )


def test_depth_parameters_use_softplus_transform_by_default():
    model = ETAS3D(
        parameters_3d(),
        maximum_depth=40.0,
        background_depth_concentration=6.0,
    )

    assert model.eta_parameter.transform == "softplus"
    assert model.delta_parameter.transform == "softplus"
    assert torch.isclose(model.eta, torch.tensor(8.0))
    assert torch.isclose(model.delta, torch.tensor(0.2))


def test_depth_parameter_transform_can_be_overridden():
    parameters = parameters_3d()
    parameters = ETAS3DParameters(
        **{
            **parameters.__dict__,
            "eta": ParameterConfig(8.0, transform="square"),
            "delta": ParameterConfig(0.2, transform="square"),
        }
    )
    model = ETAS3D(
        parameters,
        maximum_depth=40.0,
        background_depth_concentration=6.0,
    )

    assert model.eta_parameter.transform == "square"
    assert model.delta_parameter.transform == "square"
    assert torch.isclose(model.eta_parameter.raw, torch.sqrt(torch.tensor(8.0)))
    assert torch.isclose(model.delta_parameter.raw, torch.sqrt(torch.tensor(0.2)))
    assert torch.isclose(model.eta, torch.tensor(8.0))
    assert torch.isclose(model.delta, torch.tensor(0.2))


def test_reference_magnitude_parameterization_is_exactly_equivalent():
    original = ETAS3D(
        parameters_3d(),
        maximum_depth=40.0,
        background_depth_concentration=6.0,
    )
    centered = ETAS3D(
        parameters_3d(),
        maximum_depth=40.0,
        background_depth_concentration=6.0,
        depth_reference_magnitude_offset=1.0,
    )

    source = Catalog3DStub(
        time=torch.tensor([0.0, 1.0]),
        position=torch.tensor([[0.0, 0.0], [0.5, 0.5]]),
        depth=torch.tensor([8.0, 12.0]),
        magnitude=torch.tensor([4.5, 5.0]),
        completeness_magnitude=4.0,
    )
    expected_reference = original.eta * torch.exp(-original.delta)

    assert torch.isclose(
        centered.depth_reference_concentration, expected_reference
    )
    assert torch.isclose(centered.eta, original.eta)
    assert torch.allclose(
        centered.depth_concentration(source), original.depth_concentration(source)
    )


def test_recentered_reference_round_trips_through_state_dict():
    source_model = ETAS3D(
        parameters_3d(),
        maximum_depth=40.0,
        background_depth_concentration=6.0,
        depth_reference_magnitude_offset=1.0,
    )
    source_model.recenter_depth_reference_magnitude_offset(0.7)
    restored_model = ETAS3D(
        parameters_3d(),
        maximum_depth=40.0,
        background_depth_concentration=6.0,
        depth_reference_magnitude_offset=1.0,
    )
    restored_model.load_state_dict(source_model.state_dict())

    assert abs(restored_model.depth_reference_magnitude_offset - 0.7) < 1e-6
    assert torch.isclose(restored_model.eta, source_model.eta)
    assert torch.isclose(
        restored_model.depth_reference_concentration,
        source_model.depth_reference_concentration,
    )


def test_zero_depth_concentration_is_the_uniform_beta_limit():
    target = torch.tensor([0.0, 10.0, 40.0], dtype=torch.float64)
    source = torch.tensor([20.0], dtype=torch.float64)

    density = conditional_beta_depth_density(
        target[:, None],
        source[None, :],
        40.0,
        torch.zeros((1, 1), dtype=torch.float64),
    )

    assert torch.allclose(density, torch.full_like(density, 1.0 / 40.0))


def make_problem(dtype=torch.float64):
    catalog = Catalog3DStub(
        time=torch.tensor([-1.0, 0.5, 1.0, 2.0, 4.5], dtype=dtype),
        position=torch.tensor([
            [0.2, 0.2], [0.4, 0.4], [0.6, 0.5], [1.0, 1.0], [1.5, 1.5]
        ], dtype=dtype),
        depth=torch.tensor([5.0, 10.0, 15.0, 20.0, 30.0], dtype=dtype),
        magnitude=torch.tensor([3.0, 3.2, 3.1, 3.4, 3.0], dtype=dtype),
        completeness_magnitude=3.0,
    )
    domain = DomainStub(
        time_zero=-1.0,
        time_start=0.0,
        time_end=5.0,
        border=torch.tensor([[0., 0.], [2., 0.], [2., 2.], [0., 2.]], dtype=dtype),
    )
    infinite = InfiniteSpaceIntegrator()
    background = GaussianBetaKDEBackground(
        maximum_depth=40.0,
        depth_concentration=6.0,
        target_chunk_size=2,
        source_chunk_size=2,
        spatial_integrator=infinite,
    )
    model = ETAS3D(
        parameters_3d(),
        maximum_depth=40.0,
        background_depth_concentration=6.0,
        background=background,
        spatial_integrator=infinite,
    ).to(dtype=dtype)
    state = KDEState(
        position=catalog.position.clone(),
        depth=catalog.depth.clone(),
        bandwidth=torch.full((5,), 0.5, dtype=dtype),
        weights=torch.tensor([0.3, 0.2, 0.1, 0.25, 0.15], dtype=dtype),
        duration=5.0,
    )
    return model, catalog, domain, state


def test_depth_density_matches_fortran_formula_and_boundary_correction():
    dtype = torch.float64
    target = torch.tensor([[10.0], [0.0], [40.0]], dtype=dtype)
    source = torch.tensor([[8.0, 24.0]], dtype=dtype)
    concentration = torch.tensor([[6.0, 3.0]], dtype=dtype)
    actual = conditional_beta_depth_density(
        target, source, 40.0, concentration, boundary_offset=0.5
    )

    target_fraction = torch.tensor([[0.25], [0.5 / 40.0], [1.0 - 0.5 / 40.0]], dtype=dtype)
    source_fraction = source / 40.0
    a = concentration * source_fraction + 1.0
    b = concentration * (1.0 - source_fraction) + 1.0
    expected = torch.exp(
        (a - 1.0) * torch.log(target_fraction)
        + (b - 1.0) * torch.log1p(-target_fraction)
        + torch.lgamma(a + b) - torch.lgamma(a) - torch.lgamma(b)
    ) / 40.0
    assert torch.allclose(actual, expected, rtol=1e-13, atol=1e-13)


def test_depth_density_integrates_to_one():
    dtype = torch.float64
    target = torch.linspace(1e-5, 40.0 - 1e-5, 100001, dtype=dtype)[:, None]
    source = torch.tensor([[5.0, 20.0, 35.0]], dtype=dtype)
    concentration = torch.tensor([[2.0, 8.0, 15.0]], dtype=dtype)
    density = conditional_beta_depth_density(target, source, 40.0, concentration)
    mass = torch.trapezoid(density, target[:, 0], dim=0)
    assert torch.allclose(mass, torch.ones_like(mass), atol=2e-5, rtol=2e-5)


def test_pairwise_rate_is_2d_rate_times_depth_density():
    model3d, catalog, _, _ = make_problem()
    base = parameters_3d()
    model2d = ETAS2D(
        ETAS2DParameters(
            mu=base.mu, amplitude=base.amplitude, alpha=base.alpha,
            c=base.c, p=base.p, d=base.d, q=base.q, gamma=base.gamma,
        ),
        spatial_integrator=InfiniteSpaceIntegrator(),
    ).to(dtype=catalog.time.dtype)
    target = catalog.take(slice(2, 5))
    source = catalog.take(slice(0, 3))
    expected = model2d.pairwise_rate(target, source) * model3d.depth_kernel(target, source)
    assert torch.allclose(model3d.pairwise_rate(target, source), expected)


def test_full_depth_compensator_matches_horizontal_compensator():
    model3d, catalog, domain, _ = make_problem()
    base = parameters_3d()
    model2d = ETAS2D(
        ETAS2DParameters(
            mu=base.mu, amplitude=base.amplitude, alpha=base.alpha,
            c=base.c, p=base.p, d=base.d, q=base.q, gamma=base.gamma,
        ),
        spatial_integrator=InfiniteSpaceIntegrator(),
    ).to(dtype=catalog.time.dtype)
    assert torch.allclose(
        model3d.integrated_source(catalog, domain),
        model2d.integrated_source(catalog, domain),
    )


def test_gaussian_beta_background_matches_direct_product_formula():
    model, catalog, _, state = make_problem()
    actual = model.background.evaluate(catalog, state)
    displacement = catalog.position[:, None, :] - state.position[None, :, :]
    r2 = displacement.square().sum(dim=-1)
    h2 = state.bandwidth.square()[None, :]
    horizontal = torch.exp(-r2 / (2.0 * h2)) / (2.0 * torch.pi * h2)
    depth = conditional_beta_depth_density(
        catalog.depth[:, None], state.depth[None, :], 40.0,
        catalog.depth.new_tensor(6.0), boundary_offset=0.5,
    )
    expected = (horizontal * depth * state.weights[None, :]).sum(dim=-1) / state.duration
    assert torch.allclose(actual, expected)


def test_exact_likelihood_has_finite_gradients_for_all_ten_parameters():
    model, catalog, domain, state = make_problem()
    likelihood = ExactLikelihood(target_chunk_size=2, source_chunk_size=2)
    loss = likelihood(model, catalog, domain, state)
    loss.backward()
    assert torch.isfinite(loss)
    assert len(model.parameter_values()) == 10
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad), name
