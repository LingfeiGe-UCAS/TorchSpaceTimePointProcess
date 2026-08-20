from typing import Any

import torch


class PolygonSpatialIntegrator:
    """Signed-sector Simpson integration for any radial 2-D kernel."""

    def __init__(self, subdivisions: int = 1001, source_chunk_size: int = 4096):
        if subdivisions < 2:
            raise ValueError("subdivisions must be at least 2")
        if source_chunk_size < 1:
            raise ValueError("source_chunk_size must be positive")
        self.subdivisions = subdivisions
        self.source_chunk_size = source_chunk_size

    def integrate(self, kernel: Any, source: Any, domain: Any) -> torch.Tensor:
        if not hasattr(domain, "border"):
            raise TypeError("domain must provide border with shape [N_vertices, 2]")
        border = torch.as_tensor(
            domain.border, dtype=source.position.dtype, device=source.position.device
        )
        if border.ndim != 2 or border.shape[0] < 3 or border.shape[1] != 2:
            raise ValueError("domain.border must have shape [N_vertices, 2]")
        if source.position.ndim != 2 or source.position.shape[1] != 2:
            raise ValueError("source.position must have shape [N_source, 2]")

        closed = torch.cat((border, border[:1]), dim=0)
        signed_twice_area = torch.sum(
            closed[:-1, 0] * closed[1:, 1]
            - closed[1:, 0] * closed[:-1, 1]
        )
        if signed_twice_area < 0:
            border = torch.flip(border, dims=(0,))

        source_count = source.position.shape[0]
        result = source.position.new_zeros(source_count)
        fractions = torch.linspace(
            0.0, 1.0, self.subdivisions,
            dtype=source.position.dtype, device=source.position.device,
        )

        for begin in range(0, source_count, self.source_chunk_size):
            end = min(begin + self.source_chunk_size, source_count)
            position = source.position[begin:end]
            source_chunk = _SourceChunk(source, begin, end, source_count)
            mass = source.position.new_zeros(end - begin)

            for edge_index in range(border.shape[0]):
                edge_start = border[edge_index]
                edge_end = border[(edge_index + 1) % border.shape[0]]
                points = edge_start + fractions[:, None] * (edge_end - edge_start)
                first, second = points[:-1], points[1:]

                vector_first = first[None, :, :] - position[:, None, :]
                vector_second = second[None, :, :] - position[:, None, :]
                radius_first = torch.linalg.vector_norm(vector_first, dim=-1)
                radius_second = torch.linalg.vector_norm(vector_second, dim=-1)
                cross = (
                    vector_first[..., 0] * vector_second[..., 1]
                    - vector_first[..., 1] * vector_second[..., 0]
                )
                dot = torch.sum(vector_first * vector_second, dim=-1)
                angle = torch.atan2(cross, dot)

                weight = radius_first / (radius_first + radius_second).clamp_min(
                    torch.finfo(radius_first.dtype).eps
                )
                middle = first[None, :, :] + weight[..., None] * (
                    second - first
                )[None, :, :]
                radius_middle = torch.linalg.vector_norm(
                    middle - position[:, None, :], dim=-1
                )

                primitive_first = kernel.spatial_angular_integral(
                    radius_first, source_chunk
                )
                primitive_middle = kernel.spatial_angular_integral(
                    radius_middle, source_chunk
                )
                primitive_second = kernel.spatial_angular_integral(
                    radius_second, source_chunk
                )
                mass = mass + (
                    angle
                    * (
                        primitive_first / 6.0
                        + 2.0 * primitive_middle / 3.0
                        + primitive_second / 6.0
                    )
                ).sum(dim=1)
            result[begin:end] = mass
        return result


class _SourceChunk:
    """A generic leading-dimension view of a kernel source object."""

    def __init__(self, source: Any, begin: int, end: int, source_count: int):
        for name, value in vars(source).items():
            if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == source_count:
                value = value[begin:end]
            setattr(self, name, value)
