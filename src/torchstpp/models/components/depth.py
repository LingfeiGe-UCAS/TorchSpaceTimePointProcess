import torch
from .fixed_inputs import FixedInputCache


_depth_checks = FixedInputCache()


def _check_depth(value, maximum, role):
    def check():
        if torch.any(value < 0) or torch.any(value > maximum):
            raise ValueError(f"{role} depth must lie in [0, maximum_depth]")
    if isinstance(maximum, (float, int)):
        _depth_checks.get_or_create((value,), (maximum, role), check)
    else:
        check()


def conditional_beta_depth_density(
    target_depth: torch.Tensor,
    source_depth: torch.Tensor,
    maximum_depth: float | torch.Tensor,
    concentration: torch.Tensor,
    *,
    boundary_offset: float | None = 0.5,
) -> torch.Tensor:
    """Conditional Beta depth density used by the 3-D point-source ETAS model.

    This implements the point-source depth factor in the Fortran code:
    ``dbeta(z/Z, H*z_source/Z + 1, H*(1-z_source/Z) + 1) / Z``.
    A positive ``boundary_offset`` follows the legacy triggering-kernel
    treatment of events exactly at zero or at the maximum seismogenic depth.
    Passing ``None`` retains exact endpoints, as the legacy background KDE
    does; endpoint products are evaluated with ``xlogy`` so ``0 * log(0)``
    has its mathematically correct value instead of producing NaN.
    """
    if not target_depth.is_floating_point() or not source_depth.is_floating_point():
        raise TypeError("depth tensors must use a floating-point dtype")
    maximum = torch.as_tensor(
        maximum_depth, dtype=target_depth.dtype, device=target_depth.device
    )
    maximum_check = maximum_depth if isinstance(maximum_depth, (float, int)) else maximum
    if maximum.ndim != 0 or maximum_check <= 0:
        raise ValueError("maximum_depth must be a positive scalar")
    offset = None
    if boundary_offset is not None:
        offset = torch.as_tensor(
            boundary_offset, dtype=target_depth.dtype, device=target_depth.device
        )
        offset_check = boundary_offset if isinstance(boundary_offset, (float, int)) else offset
        if offset.ndim != 0 or offset_check <= 0 or 2.0 * offset_check >= maximum_check:
            raise ValueError(
                "boundary_offset must be None or a positive value smaller "
                "than maximum_depth / 2"
            )
    _check_depth(target_depth, maximum_check, "target")
    _check_depth(source_depth, maximum_check, "source")
    if torch.any(~torch.isfinite(concentration)) or torch.any(concentration < 0):
        raise ValueError("depth concentration must be finite and non-negative")

    target_fraction = target_depth / maximum
    if offset is not None:
        boundary_fraction = offset / maximum
        target_fraction = torch.where(
            target_depth <= 0,
            boundary_fraction,
            torch.where(
                target_depth >= maximum,
                1.0 - boundary_fraction,
                target_fraction,
            ),
        )
    source_fraction = source_depth / maximum
    shape_a = concentration * source_fraction + 1.0
    shape_b = concentration * (1.0 - source_fraction) + 1.0
    log_density = (
        torch.xlogy(shape_a - 1.0, target_fraction)
        + torch.xlogy(shape_b - 1.0, 1.0 - target_fraction)
        + torch.lgamma(shape_a + shape_b)
        - torch.lgamma(shape_a)
        - torch.lgamma(shape_b)
        - torch.log(maximum)
    )
    return torch.exp(log_density)
