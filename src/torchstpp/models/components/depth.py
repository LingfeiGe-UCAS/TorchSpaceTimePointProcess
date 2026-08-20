import torch


def conditional_beta_depth_density(
    target_depth: torch.Tensor,
    source_depth: torch.Tensor,
    maximum_depth: float | torch.Tensor,
    concentration: torch.Tensor,
    *,
    boundary_offset: float = 0.5,
) -> torch.Tensor:
    """Conditional Beta depth density used by the 3-D point-source ETAS model.

    This implements the point-source depth factor in the Fortran code:
    ``dbeta(z/Z, H*z_source/Z + 1, H*(1-z_source/Z) + 1) / Z``.
    The half-unit boundary correction follows the legacy treatment of events
    exactly at zero or at the maximum seismogenic depth.
    """
    if not target_depth.is_floating_point() or not source_depth.is_floating_point():
        raise TypeError("depth tensors must use a floating-point dtype")
    maximum = torch.as_tensor(
        maximum_depth, dtype=target_depth.dtype, device=target_depth.device
    )
    offset = torch.as_tensor(
        boundary_offset, dtype=target_depth.dtype, device=target_depth.device
    )
    if maximum.ndim != 0 or maximum <= 0:
        raise ValueError("maximum_depth must be a positive scalar")
    if offset.ndim != 0 or offset <= 0 or 2.0 * offset >= maximum:
        raise ValueError("boundary_offset must be positive and smaller than maximum_depth / 2")
    if torch.any(target_depth < 0) or torch.any(target_depth > maximum):
        raise ValueError("target depth must lie in [0, maximum_depth]")
    if torch.any(source_depth < 0) or torch.any(source_depth > maximum):
        raise ValueError("source depth must lie in [0, maximum_depth]")
    if torch.any(~torch.isfinite(concentration)) or torch.any(concentration < 0):
        raise ValueError("depth concentration must be finite and non-negative")

    target_fraction = target_depth / maximum
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
        (shape_a - 1.0) * torch.log(target_fraction)
        + (shape_b - 1.0) * torch.log1p(-target_fraction)
        + torch.lgamma(shape_a + shape_b)
        - torch.lgamma(shape_a)
        - torch.lgamma(shape_b)
        - torch.log(maximum)
    )
    return torch.exp(log_density)
