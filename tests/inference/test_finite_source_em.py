from types import SimpleNamespace

import torch

from torchstpp.finite_source.em import (
    finite_source_patch_responsibilities,
    smooth_patch_responsibilities,
    update_finite_source_weights,
)
from torchstpp.inference import LegacyEMLBFGS, LegacyEMLBFGSConfig
from torchstpp.likelihood import ExactLikelihood
from torchstpp.models.backgrounds import KDEState
from torchstpp.models.components import conditional_beta_depth_density

from tests.likelihood.test_exact import make_problem


class _Catalog:
    def __init__(
        self,
        event_id,
        time,
        position,
        magnitude,
        depth,
        finite_source_id,
        completeness_magnitude=None,
    ):
        self.event_id = event_id
        self.time = time
        self.position = position
        self.magnitude = magnitude
        self.depth = depth
        self.finite_source_id = finite_source_id
        self.completeness_magnitude = (
            time.new_tensor(0.0)
            if completeness_magnitude is None
            else completeness_magnitude
        )

    def take(self, item):
        return _Catalog(
            self.event_id[item],
            self.time[item],
            self.position[item],
            self.magnitude[item],
            self.depth[item],
            self.finite_source_id[item],
            self.completeness_magnitude,
        )


class _ConstantBackground:
    @staticmethod
    def evaluate(target, state):
        return target.time.new_ones(target.time.shape)


class _FiniteModel:
    def __init__(
        self,
        catalog,
        *,
        patch_position=None,
        patch_depth=None,
        patch_tau=None,
        base_patch_rate=None,
        normalization="corrected",
        zero_policy="retain",
    ):
        dtype = catalog.time.dtype
        self.background = _ConstantBackground()
        self.geometry = SimpleNamespace(
            position=(
                torch.tensor([[0.0, 0.0], [100.0, 0.0]], dtype=dtype)
                if patch_position is None
                else patch_position
            ),
            depth=(
                torch.tensor([10.0, 10.0], dtype=dtype)
                if patch_depth is None
                else patch_depth
            ),
        )
        self.patch_tau = (
            torch.ones(2, dtype=dtype)
            if patch_tau is None
            else patch_tau.clone()
        )
        self.patch_tau_total = self.patch_tau.sum().reshape(1)
        self.patch_smoothing_normalizer = torch.ones_like(self.patch_tau)
        self.base_patch_rate = (
            torch.tensor(
                [[0.0, 0.0], [1.0, 3.0], [2.0, 0.0]], dtype=dtype
            )
            if base_patch_rate is None
            else base_patch_rate
        )
        self.maximum_depth = 30.0
        self.depth_boundary_offset = 0.5
        self.patch_chunk_size = 1
        self.horizontal_smoothing_bandwidth = 0.01
        self.depth_smoothing_concentration = 10.0
        self.tau_normalization = normalization
        self.zero_responsibility_policy = zero_policy

    @staticmethod
    def finite_source_parent_indices(catalog):
        return torch.nonzero(catalog.finite_source_id > 0, as_tuple=False).flatten()

    @staticmethod
    def patch_slice(group):
        if group != 1:
            raise KeyError(group)
        return slice(0, 2)

    def finite_patch_rate(self, target, parent, begin, end):
        del parent
        weight = self.patch_tau / self.patch_tau_total[0]
        return self.base_patch_rate[target.event_id][:, begin:end] * weight[begin:end]

    def pairwise_rate(self, target, source):
        result = target.time.new_zeros((target.time.numel(), source.time.numel()))
        for column, event_id in enumerate(source.event_id.tolist()):
            if event_id == 0:
                result[:, column] = self.finite_patch_rate(
                    target, source.take(slice(column, column + 1)), 0, 2
                ).sum(dim=1)
        return result

    @staticmethod
    def background_rate(value):
        return value

    def set_patch_tau_(self, value):
        if value.shape != self.patch_tau.shape:
            raise ValueError
        self.patch_tau.copy_(value)
        self.patch_tau_total[0] = value.sum()


def _problem(*, patch_tau=None, base_patch_rate=None, **model_kwargs):
    dtype = torch.float64
    catalog = _Catalog(
        event_id=torch.arange(3, dtype=torch.int64),
        time=torch.tensor([0.0, 1.0, 2.0], dtype=dtype),
        position=torch.zeros((3, 2), dtype=dtype),
        magnitude=torch.ones(3, dtype=dtype),
        depth=torch.full((3,), 10.0, dtype=dtype),
        finite_source_id=torch.tensor([1, 0, 0], dtype=torch.int64),
    )
    state = KDEState(
        position=catalog.position,
        bandwidth=torch.ones(3, dtype=dtype),
        weights=torch.ones(3, dtype=dtype),
        duration=2.0,
        depth=catalog.depth,
    )
    return (
        _FiniteModel(
            catalog,
            patch_tau=patch_tau,
            base_patch_rate=base_patch_rate,
            **model_kwargs,
        ),
        catalog,
        state,
    )


def test_patch_responsibilities_use_current_tau_and_total_intensity():
    model, catalog, state = _problem()

    result = update_finite_source_weights(
        model,
        catalog,
        state,
        target_chunk_size=1,
        source_chunk_size=2,
        patch_chunk_size=1,
    )

    assert result is not None
    expected = torch.tensor([2.0 / 3.0, 0.5], dtype=torch.float64)
    assert torch.allclose(result.expected_patch_offspring, expected)
    # The two patches are separated far beyond tau_h, so smoothing is diagonal.
    assert torch.allclose(model.patch_tau, expected)


def test_disjoint_target_ranges_sum_to_the_full_patch_e_step():
    model, catalog, state = _problem()
    full = finite_source_patch_responsibilities(
        model, catalog, state, target_chunk_size=1, source_chunk_size=2
    )
    first = finite_source_patch_responsibilities(
        model,
        catalog,
        state,
        target_chunk_size=1,
        source_chunk_size=2,
        target_ranges=((0, 2),),
    )
    second = finite_source_patch_responsibilities(
        model,
        catalog,
        state,
        target_chunk_size=1,
        source_chunk_size=2,
        target_ranges=((2, 3),),
    )

    assert full is not None and first is not None and second is not None
    assert torch.allclose(first + second, full)


def test_distributed_reduction_reconstructs_global_patch_responsibilities():
    model, catalog, state = _problem()
    remote = finite_source_patch_responsibilities(
        model,
        catalog,
        state,
        target_chunk_size=1,
        source_chunk_size=2,
        target_ranges=((2, 3),),
    )
    assert remote is not None

    class _Distributed:
        @staticmethod
        def global_sum_(value):
            value.add_(remote)

    result = update_finite_source_weights(
        model,
        catalog,
        state,
        target_chunk_size=1,
        source_chunk_size=2,
        target_ranges=((0, 2),),
        distributed_likelihood=_Distributed(),
    )

    assert result is not None
    assert torch.allclose(
        result.expected_patch_offspring,
        torch.tensor([2.0 / 3.0, 0.5], dtype=torch.float64),
    )


def test_fortran_and_corrected_tau_normalizers_match_their_matrix_formulas():
    dtype = torch.float64
    position = torch.zeros((2, 2), dtype=dtype)
    depth = torch.tensor([5.0, 20.0], dtype=dtype)
    expected = torch.tensor([1.0, 2.0], dtype=dtype)
    corrected, catalog, _ = _problem(
        patch_position=position,
        patch_depth=depth,
        normalization="corrected",
    )
    fortran, _, _ = _problem(
        patch_position=position,
        patch_depth=depth,
        normalization="fortran",
    )

    corrected_result = smooth_patch_responsibilities(
        corrected, catalog, expected, patch_chunk_size=1
    )
    fortran_result = smooth_patch_responsibilities(
        fortran, catalog, expected, patch_chunk_size=1
    )
    assert corrected_result is not None and fortran_result is not None

    concentration = depth.new_tensor(10.0)
    kernel = conditional_beta_depth_density(
        depth[:, None],
        depth[None, :],
        30.0,
        concentration,
        boundary_offset=0.5,
    )
    corrected_expected = kernel @ (expected / kernel.sum(dim=0))
    fortran_expected = kernel @ (expected / kernel.sum(dim=1))
    assert torch.allclose(corrected.patch_tau, corrected_expected)
    assert torch.allclose(fortran.patch_tau, fortran_expected)
    assert not torch.allclose(corrected.patch_tau, fortran.patch_tau)


def test_zero_responsibility_fallback_can_retain_or_reset_uniform_tau():
    zero_rate = torch.zeros((3, 2), dtype=torch.float64)
    initial = torch.tensor([3.0, 1.0], dtype=torch.float64)
    retained, catalog, state = _problem(
        patch_tau=initial,
        base_patch_rate=zero_rate,
        zero_policy="retain",
    )
    uniform, _, _ = _problem(
        patch_tau=initial,
        base_patch_rate=zero_rate,
        zero_policy="uniform",
    )

    retained_result = update_finite_source_weights(retained, catalog, state)
    uniform_result = update_finite_source_weights(uniform, catalog, state)

    assert retained_result is not None and uniform_result is not None
    assert retained_result.zero_responsibility_groups == (1,)
    assert uniform_result.zero_responsibility_groups == (1,)
    assert torch.equal(retained.patch_tau, initial)
    assert torch.equal(uniform.patch_tau, torch.ones_like(initial))


def test_point_source_model_is_an_exact_no_op():
    _, catalog, state = _problem()
    model = SimpleNamespace()

    assert update_finite_source_weights(model, catalog, state) is None


def test_legacy_loop_passes_pre_update_kde_to_finite_rescale(monkeypatch):
    model, catalog, domain, state = make_problem()
    seen = []

    def record_state(model, catalog, background_state, **kwargs):
        del model, catalog, kwargs
        seen.append(background_state.weights.detach().clone())
        return None

    monkeypatch.setattr(
        "torchstpp.inference.legacy.update_finite_source_weights", record_state
    )
    fitter = LegacyEMLBFGS(
        LegacyEMLBFGSConfig(
            outer_iterations=1,
            e_steps_per_outer=1,
            decrease_e_steps=False,
            lbfgs_max_iter=1,
            line_search_fn=None,
            target_chunk_size=2,
            source_chunk_size=2,
        )
    )
    fitter.fit(
        model,
        ExactLikelihood(target_chunk_size=2, source_chunk_size=2),
        catalog,
        domain,
        state,
    )

    assert len(seen) == 1
    assert torch.equal(seen[0], state.weights)
