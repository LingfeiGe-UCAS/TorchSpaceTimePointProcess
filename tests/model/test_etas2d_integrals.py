from types import SimpleNamespace

import torch

from torchstpp.integration import InfiniteSpaceIntegrator, PolygonSpatialIntegrator
from torchstpp.models.etas import ETAS2D


def make_catalog():
    return SimpleNamespace(
        time=torch.tensor([1.0, 9.0]),
        position=torch.tensor([[0.0, 0.0], [1.0, 1.0]]),
        magnitude=torch.tensor([3.0, 4.0]),
        completeness_magnitude=3.0,
    )


def make_domain():
    return SimpleNamespace(time_start=0.0, time_end=10.0, area=20.0)


def test_etas2d_forward_and_integrals():
    catalog, domain, model = make_catalog(), make_domain(), ETAS2D()
    space_mass = InfiniteSpaceIntegrator().integrate(model, catalog, domain)
    background_values = torch.tensor([0.4, 0.6])
    background_mass = torch.tensor(1.0)

    assert model(catalog, background_values).shape == (2,)
    assert torch.allclose(
        model.background_rate(background_values), model.mu * background_values
    )
    assert model.integrated_background(background_mass).ndim == 0
    assert model.integrated_source(catalog, domain, space_mass).shape == (2,)

    radius = torch.tensor([0.0, 1.0e6])
    radial_source = SimpleNamespace(
        magnitude=torch.tensor([3.0, 3.0]), completeness_magnitude=3.0
    )
    mass = model.spatial_radial_integral(radius, radial_source)
    assert torch.isclose(mass[0], torch.tensor(0.0))
    assert mass[1] > 0.999

    loss = model.integrated_background(background_mass) + model.integrated_source(
        catalog, domain, space_mass
    ).sum()
    loss.backward()
    assert model.mu_parameter.raw.grad is not None
    assert model.p_parameter.raw.grad is not None


def test_polygon_spatial_integrator():
    catalog, model = make_catalog(), ETAS2D()
    domain = SimpleNamespace(
        border=torch.tensor([
            [-1.0e6, -1.0e6], [1.0e6, -1.0e6],
            [1.0e6, 1.0e6], [-1.0e6, 1.0e6],
        ])
    )
    mass = PolygonSpatialIntegrator(subdivisions=33).integrate(
        model, catalog, domain
    )
    assert mass.shape == catalog.time.shape
    assert torch.all(mass > 0.999)
    assert torch.all(mass <= 1.001)
    mass.sum().backward()
    assert model.d_parameter.raw.grad is not None


def test_etas2d_uses_old_polygon_method_by_default():
    catalog, model = make_catalog(), ETAS2D()
    domain = SimpleNamespace(
        time_start=0.0,
        time_end=10.0,
        area=4.0e12,
        border=torch.tensor([
            [-1.0e6, -1.0e6], [1.0e6, -1.0e6],
            [1.0e6, 1.0e6], [-1.0e6, 1.0e6],
        ]),
    )
    assert isinstance(model.spatial_integrator, PolygonSpatialIntegrator)
    result = model.integrated_source(catalog, domain)
    assert result.shape == catalog.time.shape
    assert torch.all(torch.isfinite(result))


def test_pairwise_rate_and_forward():
    catalog, model = make_catalog(), ETAS2D()
    background_values = torch.tensor([0.4, 0.6])
    pairwise = model.pairwise_rate(catalog, catalog)

    assert pairwise.shape == (2, 2)
    # An event cannot trigger itself or an earlier target event.
    assert torch.all(pairwise.triu(diagonal=0) == 0.0)
    assert pairwise[1, 0] > 0.0

    intensity = model(catalog, background_values, catalog)
    expected = model.mu * background_values + pairwise.sum(dim=-1)
    assert torch.allclose(intensity, expected)


def test_pairwise_matches_old_formula():
    model = ETAS2D()
    source = make_catalog()
    target = SimpleNamespace(
        time=torch.tensor([2.0, 10.0, 12.0]),
        position=torch.tensor([[0.5, 0.0], [1.0, 2.0], [3.0, 1.0]]),
    )
    actual = model.pairwise_rate(target, source)

    delta_time = target.time[:, None] - source.time[None, :]
    mask = delta_time > 0.0
    safe_delta = delta_time * mask
    displacement = target.position[:, None, :] - source.position[None, :, :]
    distance_squared = displacement.square().sum(dim=-1)
    magnitude = source.magnitude[None, :]
    magnitude_difference = magnitude - source.completeness_magnitude
    productivity = model.amplitude * torch.exp(model.alpha * magnitude_difference)
    temporal = (model.p - 1.0) * torch.pow(
        1.0 + safe_delta / model.c, -model.p
    ) / model.c
    scale = model.d * torch.exp(model.gamma * magnitude_difference)
    spatial = (model.q - 1.0) * torch.pow(
        1.0 + distance_squared / scale, -model.q
    ) / (torch.pi * scale)
    expected = torch.where(mask, productivity * temporal * spatial, 0.0)

    assert torch.allclose(actual, expected)
