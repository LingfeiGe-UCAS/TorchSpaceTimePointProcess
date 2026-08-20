from dataclasses import dataclass

import torch

from torchstpp.integration import InfiniteSpaceIntegrator
from torchstpp.likelihood import ExactLikelihood, LikelihoodTerms
from torchstpp.models.backgrounds import GaussianKDEBackground, KDEState
from torchstpp.models.etas import ETAS2D, ETAS2DParameters
from torchstpp.models import ParameterConfig


@dataclass
class CatalogStub:
    time: torch.Tensor
    position: torch.Tensor
    magnitude: torch.Tensor
    completeness_magnitude: float
    depth: torch.Tensor | None = None

    def take(self, index):
        return CatalogStub(
            time=self.time[index],
            position=self.position[index],
            magnitude=self.magnitude[index],
            completeness_magnitude=self.completeness_magnitude,
            depth=None if self.depth is None else self.depth[index],
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
            (position[:, 0] >= 0.0) & (position[:, 0] <= 2.0)
            & (position[:, 1] >= 0.0) & (position[:, 1] <= 2.0)
        )

    def target_mask(self, catalog):
        return (
            (catalog.time >= self.time_start)
            & (catalog.time <= self.time_end)
            & self.contains(catalog.position)
        )

    def source_mask(self, catalog):
        return (catalog.time >= self.time_zero) & (catalog.time <= self.time_end)


def make_problem(dtype=torch.float64):
    catalog = CatalogStub(
        time=torch.tensor([-1.0, 0.5, 1.0, 2.0, 4.5, 6.0], dtype=dtype),
        position=torch.tensor([
            [0.2, 0.2], [3.0, 1.0], [0.5, 0.5],
            [1.0, 1.0], [1.5, 1.5], [3.0, 3.0],
        ], dtype=dtype),
        magnitude=torch.tensor([3.0, 3.2, 3.1, 3.4, 3.0, 3.3], dtype=dtype),
        completeness_magnitude=3.0,
    )
    domain = DomainStub(
        time_zero=-1.0,
        time_start=0.0,
        time_end=6.0,
        border=torch.tensor([[0., 0.], [2., 0.], [2., 2.], [0., 2.]], dtype=dtype),
    )
    parameters = ETAS2DParameters(
        mu=ParameterConfig(0.4), amplitude=ParameterConfig(0.2),
        alpha=ParameterConfig(0.8), c=ParameterConfig(0.1),
        p=ParameterConfig(1.2), d=ParameterConfig(0.3),
        q=ParameterConfig(1.6), gamma=ParameterConfig(0.5),
    )
    infinite = InfiniteSpaceIntegrator()
    background = GaussianKDEBackground(
        target_chunk_size=16,
        source_chunk_size=16,
        spatial_integrator=infinite,
    )
    model = ETAS2D(
        parameters,
        background=background,
        spatial_integrator=infinite,
    ).to(dtype=dtype)
    state = KDEState(
        position=catalog.position.clone(),
        bandwidth=torch.full((6,), 0.5, dtype=dtype),
        weights=torch.tensor([0.3, 0.2, 0.1, 0.2, 0.15, 0.05], dtype=dtype),
        duration=6.0,
    )
    return model, catalog, domain, state


def test_exact_likelihood_terms_and_selection():
    model, catalog, domain, state = make_problem()
    likelihood = ExactLikelihood(target_chunk_size=2, source_chunk_size=2)
    terms = likelihood.terms(model, catalog, domain, state)

    assert isinstance(terms, LikelihoodTerms)
    assert terms.target_count == 3  # times 1, 2, and 4.5 inside polygon
    assert terms.negative_log_likelihood.ndim == 0
    assert torch.isfinite(terms.negative_log_likelihood)
    assert torch.allclose(likelihood(model, catalog, domain, state), terms.negative_log_likelihood)


def test_chunked_matches_dense_value_and_gradient():
    dense_model, catalog, domain, state = make_problem()
    dense = ExactLikelihood(target_chunk_size=64, source_chunk_size=64)
    dense_loss = dense(dense_model, catalog, domain, state)
    dense_loss.backward()
    dense_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in dense_model.named_parameters()
    }

    chunked_model, catalog, domain, state = make_problem()
    chunked = ExactLikelihood(target_chunk_size=1, source_chunk_size=2)
    chunked_loss = chunked(chunked_model, catalog, domain, state)
    chunked_loss.backward()

    assert torch.allclose(chunked_loss, dense_loss, rtol=1e-12, atol=1e-12)
    for name, parameter in chunked_model.named_parameters():
        assert torch.allclose(
            parameter.grad, dense_gradients[name], rtol=1e-10, atol=1e-10
        ), name


def test_optional_depth_is_validated_but_not_used_by_etas2d():
    model, catalog, domain, state = make_problem()
    catalog.depth = torch.tensor([5., 6., 7., 8., 9., 10.], dtype=catalog.time.dtype)
    likelihood = ExactLikelihood(target_chunk_size=2, source_chunk_size=2)
    loss_with_depth = likelihood(model, catalog, domain, state)

    catalog.depth = None
    loss_without_depth = likelihood(model, catalog, domain, state)
    assert torch.allclose(loss_with_depth, loss_without_depth)


def test_invalid_depth_shape_is_rejected():
    model, catalog, domain, state = make_problem()
    catalog.depth = torch.tensor([1.0, 2.0], dtype=catalog.time.dtype)
    likelihood = ExactLikelihood()
    try:
        likelihood(model, catalog, domain, state)
    except ValueError as error:
        assert "catalog.depth" in str(error)
        return
    raise AssertionError("invalid depth shape must be rejected")


def test_checkpoint_matches_standard_value_and_gradient():
    standard_model, catalog, domain, state = make_problem()
    standard = ExactLikelihood(
        target_chunk_size=1,
        source_chunk_size=2,
        checkpoint_blocks=False,
    )
    standard_loss = standard(standard_model, catalog, domain, state)
    standard_loss.backward()
    standard_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in standard_model.named_parameters()
    }

    checkpoint_model, catalog, domain, state = make_problem()
    low_memory = ExactLikelihood(
        target_chunk_size=1,
        source_chunk_size=2,
        checkpoint_blocks=True,
    )
    checkpoint_loss = low_memory(checkpoint_model, catalog, domain, state)
    checkpoint_loss.backward()

    assert torch.allclose(checkpoint_loss, standard_loss, rtol=1e-12, atol=1e-12)
    for name, parameter in checkpoint_model.named_parameters():
        assert torch.allclose(
            parameter.grad,
            standard_gradients[name],
            rtol=1e-10,
            atol=1e-10,
        ), name


def test_checkpoint_is_optional_and_defaults_off():
    assert ExactLikelihood().checkpoint_blocks is False
    assert ExactLikelihood(checkpoint_blocks=True).checkpoint_blocks is True
