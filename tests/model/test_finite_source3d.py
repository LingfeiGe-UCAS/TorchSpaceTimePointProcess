from dataclasses import dataclass
from types import SimpleNamespace

import torch

from torchstpp.finite_source import FiniteSourceGeometry
from torchstpp.integration import InfiniteSpaceIntegrator, PolygonSpatialIntegrator
from torchstpp.models import ParameterConfig
from torchstpp.models.components import conditional_beta_depth_density
from torchstpp.models.etas import (
    ETAS3D,
    ETAS3DParameters,
    FiniteSourceETAS3D,
    FiniteSourceETAS3DParameters,
)


@dataclass
class CatalogStub:
    time: torch.Tensor
    position: torch.Tensor
    depth: torch.Tensor
    magnitude: torch.Tensor
    completeness_magnitude: torch.Tensor
    finite_source_id: torch.Tensor | None = None

    def take(self, index):
        return CatalogStub(
            time=self.time[index],
            position=self.position[index],
            depth=self.depth[index],
            magnitude=self.magnitude[index],
            completeness_magnitude=self.completeness_magnitude,
            finite_source_id=(
                None
                if self.finite_source_id is None
                else self.finite_source_id[index]
            ),
        )


@dataclass
class DomainStub:
    time_start: float
    time_end: float
    border: torch.Tensor

    @property
    def area(self):
        return 4.0


def parameters() -> FiniteSourceETAS3DParameters:
    return FiniteSourceETAS3DParameters(
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
        chara_d=ParameterConfig(0.12),
        ch_eta=ParameterConfig(5.0),
    )


def point_parameters() -> ETAS3DParameters:
    values = parameters()
    return ETAS3DParameters(
        mu=values.mu,
        amplitude=values.amplitude,
        alpha=values.alpha,
        c=values.c,
        p=values.p,
        d=values.d,
        q=values.q,
        gamma=values.gamma,
        eta=values.eta,
        delta=values.delta,
    )


def geometry(*, duplicate_first: bool = True) -> FiniteSourceGeometry:
    dtype = torch.float64
    if duplicate_first:
        return FiniteSourceGeometry(
            patch_id=torch.tensor([100, 101, 200]),
            finite_source_id=torch.tensor([7, 7, 9]),
            position=torch.tensor(
                [[0.45, 0.50], [0.45, 0.50], [1.20, 1.10]], dtype=dtype
            ),
            depth=torch.tensor([11.0, 11.0, 19.0], dtype=dtype),
        )
    return FiniteSourceGeometry(
        patch_id=torch.tensor([100, 200]),
        finite_source_id=torch.tensor([7, 9]),
        position=torch.tensor([[0.45, 0.50], [1.20, 1.10]], dtype=dtype),
        depth=torch.tensor([11.0, 19.0], dtype=dtype),
    )


def make_model(
    *,
    patch_chunk_size: int = 1,
    integrator=None,
    duplicate_first: bool = True,
    patch_tau=None,
    zero_responsibility_policy: str = "retain",
) -> FiniteSourceETAS3D:
    integrator = integrator or InfiniteSpaceIntegrator()
    return FiniteSourceETAS3D(
        parameters(),
        finite_source_geometry=geometry(duplicate_first=duplicate_first),
        maximum_depth=40.0,
        background_depth_concentration=6.0,
        spatial_integrator=integrator,
        patch_chunk_size=patch_chunk_size,
        patch_tau=patch_tau,
        zero_responsibility_policy=zero_responsibility_policy,
    ).to(dtype=torch.float64)


def source_catalog(ids=(0, 7, 9)) -> CatalogStub:
    dtype = torch.float64
    return CatalogStub(
        time=torch.tensor([0.0, 0.5, 1.0], dtype=dtype),
        position=torch.tensor(
            [[0.1, 0.1], [0.5, 0.5], [1.1, 1.1]], dtype=dtype
        ),
        depth=torch.tensor([7.0, 12.0, 18.0], dtype=dtype),
        magnitude=torch.tensor([3.4, 3.8, 4.0], dtype=dtype),
        completeness_magnitude=torch.tensor(3.0, dtype=dtype),
        finite_source_id=torch.tensor(ids, dtype=torch.int64),
    )


def target_catalog() -> CatalogStub:
    dtype = torch.float64
    return CatalogStub(
        time=torch.tensor([2.0, 3.0], dtype=dtype),
        position=torch.tensor([[0.6, 0.55], [1.4, 1.3]], dtype=dtype),
        depth=torch.tensor([13.0, 21.0], dtype=dtype),
        magnitude=torch.tensor([3.1, 3.2], dtype=dtype),
        completeness_magnitude=torch.tensor(3.0, dtype=dtype),
        finite_source_id=torch.zeros(2, dtype=torch.int64),
    )


def test_parameters_and_group_normalization_are_exposed():
    model = make_model(patch_tau=torch.tensor([1.0, 3.0, 2.0]))

    assert list(model.parameter_values())[-2:] == ["chara_d", "ch_eta"]
    assert torch.isclose(model.chara_d, torch.tensor(0.12, dtype=torch.float64))
    assert torch.isclose(model.ch_eta, torch.tensor(5.0, dtype=torch.float64))
    assert model.patch_slice(7) == slice(0, 2)
    assert model.patch_slice(torch.tensor(9)) == slice(2, 3)
    assert torch.allclose(
        model.normalized_patch_tau,
        torch.tensor([0.25, 0.75, 1.0], dtype=torch.float64),
    )


def test_all_point_catalog_is_exactly_the_existing_etas3d_path():
    finite = make_model()
    point = ETAS3D(
        point_parameters(),
        maximum_depth=40.0,
        background_depth_concentration=6.0,
        spatial_integrator=InfiniteSpaceIntegrator(),
    ).to(dtype=torch.float64)
    sources = source_catalog(ids=(0, 0, 0))
    targets = target_catalog()
    domain = DomainStub(
        time_start=0.0,
        time_end=4.0,
        border=torch.tensor(
            [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]],
            dtype=torch.float64,
        ),
    )

    assert torch.allclose(
        finite.pairwise_rate(targets, sources),
        point.pairwise_rate(targets, sources),
    )
    assert torch.allclose(
        finite.integrated_source(sources, domain),
        point.integrated_source(sources, domain),
    )


def test_duplicate_identical_patches_do_not_multiply_parent_productivity():
    model = make_model(patch_tau=torch.tensor([1.0, 3.0, 1.0]))
    parent = source_catalog().take(torch.tensor([1]))
    target = target_catalog()
    actual = model.pairwise_rate(target, parent)[:, 0]

    delta_time = target.time - parent.time[0]
    productivity = model.amplitude * torch.exp(
        model.alpha * (parent.magnitude[0] - parent.completeness_magnitude)
    )
    displacement = target.position - model.patch_position[0]
    distance_squared = displacement.square().sum(dim=-1)
    spatial = (model.q - 1.0) * torch.pow(
        1.0 + distance_squared / model.chara_d, -model.q
    ) / (torch.pi * model.chara_d)
    depth = conditional_beta_depth_density(
        target.depth,
        model.patch_depth[0].expand_as(target.depth),
        40.0,
        model.ch_eta,
    )
    expected = productivity * model.temporal_kernel(delta_time) * spatial * depth

    assert torch.allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_mixed_pairwise_columns_preserve_physical_source_order():
    model = make_model(patch_tau=torch.tensor([1.0, 3.0, 2.0]))
    sources = source_catalog()
    targets = target_catalog()
    mixed = model.pairwise_rate(targets, sources)

    columns = [
        model.pairwise_rate(targets, sources.take(torch.tensor([index])))[:, 0]
        for index in range(3)
    ]
    assert torch.allclose(mixed, torch.stack(columns, dim=1))
    assert model.finite_source_parent_indices(sources).tolist() == [1, 2]


def test_infinite_space_compensator_keeps_one_productivity_per_parent():
    model = make_model(patch_tau=torch.tensor([1.0, 3.0, 2.0]))
    sources = source_catalog()
    domain = DomainStub(
        time_start=0.0,
        time_end=4.0,
        border=torch.tensor(
            [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]],
            dtype=torch.float64,
        ),
    )
    expected = model.magnitude_productivity(sources) * model.temporal_integral(
        sources.time, domain.time_start, domain.time_end
    )

    assert torch.allclose(
        model.integrated_source(sources, domain), expected, rtol=1e-12, atol=1e-12
    )


def test_polygon_compensator_is_centered_on_patches_not_parent_centroid():
    integrator = PolygonSpatialIntegrator(subdivisions=101, source_chunk_size=2)
    model = make_model(
        integrator=integrator,
        duplicate_first=False,
        patch_chunk_size=1,
    )
    parent = source_catalog().take(torch.tensor([1]))
    # Put the physical-event centroid well outside; its patch remains in-domain.
    parent.position[0] = torch.tensor([20.0, 20.0], dtype=torch.float64)
    domain = DomainStub(
        time_start=0.0,
        time_end=4.0,
        border=torch.tensor(
            [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]],
            dtype=torch.float64,
        ),
    )
    patch_source = SimpleNamespace(
        time=parent.time,
        position=model.patch_position[:1],
        depth=model.patch_depth[:1],
        magnitude=parent.magnitude,
        completeness_magnitude=parent.completeness_magnitude,
        finite_source_id=torch.tensor([7]),
    )
    patch_mass = integrator.integrate(model, patch_source, domain)
    expected = (
        model.magnitude_productivity(parent)
        * model.temporal_integral(parent.time, 0.0, 4.0)
        * patch_mass
    )

    assert torch.allclose(model.integrated_source(parent, domain), expected)


def test_patch_chunking_is_value_and_gradient_invariant():
    small = make_model(patch_chunk_size=1, patch_tau=torch.tensor([1.0, 3.0, 2.0]))
    large = make_model(patch_chunk_size=99, patch_tau=torch.tensor([1.0, 3.0, 2.0]))
    large.load_state_dict(small.state_dict())
    sources = source_catalog()
    targets = target_catalog()
    domain = DomainStub(
        time_start=0.0,
        time_end=4.0,
        border=torch.tensor(
            [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]],
            dtype=torch.float64,
        ),
    )

    small_loss = (
        small.pairwise_rate(targets, sources).sum()
        + small.integrated_source(sources, domain).sum()
        + small.mu
    )
    large_loss = (
        large.pairwise_rate(targets, sources).sum()
        + large.integrated_source(sources, domain).sum()
        + large.mu
    )
    small_loss.backward()
    large_loss.backward()

    assert torch.allclose(small_loss, large_loss, rtol=1e-12, atol=1e-12)
    for (small_name, small_parameter), (large_name, large_parameter) in zip(
        small.named_parameters(), large.named_parameters()
    ):
        assert small_name == large_name
        assert small_parameter.grad is not None, small_name
        assert torch.isfinite(small_parameter.grad).all(), small_name
        assert torch.allclose(
            small_parameter.grad, large_parameter.grad, rtol=1e-11, atol=1e-12
        ), small_name


def test_tau_update_applies_configured_zero_group_policy():
    retain = make_model(patch_tau=torch.tensor([2.0, 3.0, 4.0]))
    retain.set_patch_tau_(torch.tensor([0.0, 0.0, 5.0]))
    assert torch.allclose(
        retain.patch_tau, torch.tensor([2.0, 3.0, 5.0], dtype=torch.float64)
    )
    assert torch.allclose(
        retain.patch_tau_total, torch.tensor([5.0, 5.0], dtype=torch.float64)
    )

    uniform = make_model(
        patch_tau=torch.tensor([2.0, 3.0, 4.0]),
        zero_responsibility_policy="uniform",
    )
    uniform.update_patch_tau_(torch.zeros(3))
    assert torch.allclose(uniform.patch_tau, torch.ones(3, dtype=torch.float64))
    assert torch.allclose(
        uniform.patch_tau_total, torch.tensor([2.0, 1.0], dtype=torch.float64)
    )


def test_unknown_or_reused_finite_source_parent_id_is_rejected():
    model = make_model()
    unknown = source_catalog(ids=(0, 7, 10))
    reused = source_catalog(ids=(0, 7, 7))

    try:
        model.finite_source_parent_indices(unknown)
    except KeyError as error:
        assert "without patches" in str(error)
    else:
        raise AssertionError("unknown finite-source ID was accepted")

    try:
        model.finite_source_parent_indices(reused)
    except ValueError as error:
        assert "one parent event" in str(error)
    else:
        raise AssertionError("reused finite-source ID was accepted")
