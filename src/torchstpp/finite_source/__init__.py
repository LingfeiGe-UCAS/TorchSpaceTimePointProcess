"""Finite-source point-process support."""

from .geometry import FiniteSourceGeometry
from .io import load_legacy_finite_source_geometry

__all__ = ["FiniteSourceGeometry", "load_legacy_finite_source_geometry"]
