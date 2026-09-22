# The MIT License (MIT)
# Copyright (c) 2025 by the xcube development team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NON INFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from collections.abc import Sequence
from typing import Any

import dask.array as da
import numpy as np
import xarray as xr

from .constants import FillValues, FloatInt
from .utils import _get_fill_value, _grid_spacing


def mosaic_datasets(
    datasets: Sequence[xr.Dataset],
    *,
    x_dim: str = "x",
    y_dim: str = "y",
    fill_values: FillValues | None = None,
    tile_size: tuple[int, int] | None = None,
) -> xr.Dataset:
    """Create a 2-D spatial mosaic from multiple datasets.

    The datasets must use the same regular spatial grid. Tiles may overlap.
    In overlapping regions, the first non-fill value encountered in
    ``datasets`` is retained.

    Args:
        datasets: Input datasets. All datasets must contain `x_dim` and
            `y_dim` coordinates with the same spatial resolution.
        x_dim: Name of the horizontal coordinate.
        y_dim: Name of the vertical coordinate.
        fill_values: Optional fill value(s). Can be a single value or dictionary
            by variable or type. If not set, defaults are:

            - float: NaN
            - uint8: 255
            - uint16: 65535
            - other integers: -1

        tile_size: Optional spatial output tile size as `(x, y)`.
            Defaults to the chunk size of the first dataset, or its spatial
            dimensions when the input datasets are not Dask-backed.

    Returns:
        A Dask-backed dataset when at least one input is Dask-backed;
        otherwise, a computed dataset backed by NumPy arrays.

    Raises:
        ValueError: If datasets use incompatible grids, dimensions, or variables.
    """
    if not datasets:
        raise ValueError("At least one dataset is required.")

    if tile_size is not None and (tile_size[0] <= 0 or tile_size[1] <= 0):
        raise ValueError("Chunk sizes must be positive.")

    for i, ds in enumerate(datasets):
        if x_dim not in ds.coords or y_dim not in ds.coords:
            raise ValueError(
                f"Dataset {i} must contain coordinates {x_dim!r} and {y_dim!r}."
            )
        if ds[x_dim].ndim != 1 or ds[y_dim].ndim != 1:
            raise ValueError(
                f"Only 1-D {x_dim!r} and {y_dim!r} coordinates are supported. "
                f"Please check dataset {i}"
            )
        if ds.sizes[x_dim] < 2 or ds.sizes[y_dim] < 2:
            raise ValueError(
                f"In dataset {i}, {x_dim!r} and {y_dim!r} coordinates "
                f"must contain at least two values."
            )

    # Normalize all coordinates to increasing x and decreasing y order.
    datasets = [_normalize_dataset(ds, x_dim=x_dim, y_dim=y_dim) for ds in datasets]

    # Make sure every variable exists in every dataset,
    # all datasets have the same grid spacing.
    first = datasets[0]
    is_dask_backed = any(
        isinstance(data.data, da.Array)
        for ds in datasets
        for data in ds.data_vars.values()
    )
    if tile_size is None:
        if is_dask_backed:
            tile_size = (
                first.chunksizes.get(x_dim, (first.sizes[x_dim],))[0],
                first.chunksizes.get(y_dim, (first.sizes[y_dim],))[0],
            )
        else:
            tile_size = (first.sizes[x_dim], first.sizes[y_dim])
    x0 = np.asarray(first[x_dim].values)
    y0 = np.asarray(first[y_dim].values)
    dx = _grid_spacing(x0, x_dim)
    dy = _grid_spacing(y0, y_dim)
    variables = list(first.data_vars)
    for i, ds in enumerate(datasets[1:], start=1):
        missing = set(variables) - set(ds.data_vars)
        if missing:
            raise ValueError(
                f"Dataset {i} is missing data variables: {sorted(missing)}"
            )
        _validate_grid(ds[x_dim].values, dx, x_dim)
        _validate_grid(ds[y_dim].values, dy, y_dim)

    # Determine global mosaic extent.
    x_min = min(float(ds[x_dim].values[0]) for ds in datasets)
    x_max = max(float(ds[x_dim].values[-1]) for ds in datasets)
    y_max = max(float(ds[y_dim].values[0]) for ds in datasets)
    y_min = min(float(ds[y_dim].values[-1]) for ds in datasets)
    nx = _grid_size(x_min, x_max, dx, x_dim)
    ny = _grid_size(y_min, y_max, abs(dy), y_dim)
    x = x_min + np.arange(nx) * dx
    y = y_max - np.arange(ny) * abs(dy)

    # Determine the position of every tile in the global grid.
    tile_specs = []
    for ds in datasets:
        tile_x = np.asarray(ds[x_dim].values)
        tile_y = np.asarray(ds[y_dim].values)
        x_start = _grid_index(tile_x[0], x_min, dx, x_dim)
        y_start = _grid_index(tile_y[0], y_max, dy, y_dim)
        x_stop = x_start + len(tile_x)
        y_stop = y_start + len(tile_y)
        tile_specs.append(
            {
                "dataset": ds,
                "x_start": x_start,
                "x_stop": x_stop,
                "y_start": y_start,
                "y_stop": y_stop,
            }
        )

    # Build each data variable independently.
    result = {}
    for name in variables:
        name = str(name)
        result[name] = _mosaic_variable(
            tile_specs,
            name=name,
            x=x,
            y=y,
            x_dim=x_dim,
            y_dim=y_dim,
            fill_values=fill_values,
            tile_size=tile_size,
        )

    result_ds = xr.Dataset(result)
    return result_ds if is_dask_backed else result_ds.compute()


def _mosaic_variable(
    tile_specs: list[dict[str, Any]],
    *,
    name: str,
    x: np.ndarray,
    y: np.ndarray,
    x_dim: str,
    y_dim: str,
    fill_values: FillValues,
    tile_size: tuple[int, int],
) -> xr.DataArray:
    """Build one mosaicked data variable."""

    first = tile_specs[0]["dataset"][name]

    # Move spatial dimensions to the end. This makes handling the spatial
    # block considerably simpler while preserving other dimensions.
    other_dims = [dim for dim in first.dims if dim not in (y_dim, x_dim)]
    first = first.transpose(*other_dims, y_dim, x_dim)

    # Check that all tiles have compatible non-spatial dimensions.
    for spec in tile_specs:
        data = spec["dataset"][name]
        data = data.transpose(*other_dims, y_dim, x_dim)

        if data.dims != first.dims:
            raise ValueError(
                f"Variable {name!r} has incompatible dimensions between tiles."
            )

        for dim in other_dims:
            if data.sizes[dim] != first.sizes[dim]:
                raise ValueError(
                    f"Variable {name!r} has incompatible size for "
                    f"dimension {dim!r}."
                )

    # Determine the actual missing value used by the output.
    fill_value = _get_fill_value(fill_values, name, first)

    # Construct output chunk boundaries.
    x_chunks = _chunk_boundaries(len(x), tile_size[0])
    y_chunks = _chunk_boundaries(len(y), tile_size[1])

    block_rows = []
    for y0, y1 in y_chunks:
        block_row = []
        for x0, x1 in x_chunks:
            overlapping_specs = [
                spec
                for spec in tile_specs
                if (
                    spec["x_start"] < x1
                    and spec["x_stop"] > x0
                    and spec["y_start"] < y1
                    and spec["y_stop"] > y0
                )
            ]
            block = _make_output_block(
                overlapping_specs,
                name=name,
                y0=y0,
                y1=y1,
                x0=x0,
                x1=x1,
                fill_value=fill_value,
                y_dim=y_dim,
                x_dim=x_dim,
                ref_ds=tile_specs[0]["dataset"],
            )
            block_row.append(block)
        block_rows.append(block_row)

    # `da.block` combines the independently constructed output blocks.
    mosaic = da.block(block_rows)

    # Add non-spatial dimensions back to the DataArray.
    coords = {dim: first.coords[dim] for dim in other_dims}
    coords[y_dim] = y
    coords[x_dim] = x

    return xr.DataArray(
        mosaic,
        dims=(*other_dims, y_dim, x_dim),
        coords=coords,
        name=name,
        attrs=first.attrs,
    )


def _make_output_block(
    tile_specs: list[dict[str, Any]],
    *,
    name: str,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    fill_value: FloatInt,
    y_dim: str,
    x_dim: str,
    ref_ds: xr.Dataset,
) -> da.Array:
    """Construct one spatial output block."""

    height = y1 - y0
    width = x1 - x0

    ref_array = ref_ds[name]
    other_dims = [dim for dim in ref_array.dims if dim not in (y_dim, x_dim)]
    other_shape = tuple(ref_array.sizes[dim] for dim in other_dims)
    block_shape = other_shape + (height, width)

    # One chunk representing this output block.
    result = da.full(
        block_shape,
        fill_value,
        dtype=ref_array.dtype,
        chunks=block_shape,
    )

    for spec in tile_specs:
        # Determine intersection between tile and output block.
        iy0 = max(y0, spec["y_start"])
        iy1 = min(y1, spec["y_stop"])
        ix0 = max(x0, spec["x_start"])
        ix1 = min(x1, spec["x_stop"])

        data = spec["dataset"][name].transpose(*other_dims, y_dim, x_dim)

        # Extract ONLY the overlapping portion of the tile.
        tile_y0 = iy0 - spec["y_start"]
        tile_y1 = iy1 - spec["y_start"]
        tile_x0 = ix0 - spec["x_start"]
        tile_x1 = ix1 - spec["x_start"]
        local = data.isel(
            {
                y_dim: slice(tile_y0, tile_y1),
                x_dim: slice(tile_x0, tile_x1),
            }
        ).data

        # Output-block coordinates.
        result_y0 = iy0 - y0
        result_y1 = iy1 - y0
        result_x0 = ix0 - x0
        result_x1 = ix1 - x0

        result_region = result[
            ...,
            result_y0:result_y1,
            result_x0:result_x1,
        ]

        # If result contains the fill value, take the value from local.
        # Otherwise, retain result.
        if np.isnan(fill_value):
            result_valid = ~da.isnan(result_region)
        else:
            result_valid = result_region != fill_value
        result[
            ...,
            result_y0:result_y1,
            result_x0:result_x1,
        ] = da.where(
            result_valid,
            result_region,
            local,
        )

    return result.rechunk(block_shape)


def _validate_grid(
    values: np.ndarray,
    spacing: float,
    name: str,
) -> None:
    """Validate that a coordinate has the expected spacing."""
    values = np.asarray(values)

    if len(values) < 2:
        raise ValueError(f"{name} coordinates must contain at least two values.")

    diffs = np.diff(values)
    if not np.allclose(diffs, spacing):
        raise ValueError(f"Coordinate {name!r} has incompatible grid spacing.")


def _grid_index(
    value: float,
    origin: float,
    spacing: float,
    name: str,
) -> int:
    """Convert a coordinate value to an integer grid index."""
    index = (value - origin) / spacing
    rounded = round(index)

    if not np.isclose(index, rounded):
        raise ValueError(
            f"Coordinate {value!r} of {name!r} does not lie on the " "common grid."
        )

    return int(rounded)


def _grid_size(
    minimum: float,
    maximum: float,
    spacing: float,
    name: str,
) -> int:
    """Determine the number of grid points between two coordinates."""
    size = (maximum - minimum) / spacing
    rounded = round(size)

    if not np.isclose(size, rounded):
        raise ValueError(f"Global {name!r} extent is incompatible with the grid.")

    return rounded + 1


def _chunk_boundaries(
    size: int,
    tile_size: int,
) -> list[tuple[int, int]]:
    """Create half-open chunk boundaries."""
    return [
        (start, min(start + tile_size, size)) for start in range(0, size, tile_size)
    ]


def _normalize_dataset(
    ds: xr.Dataset,
    *,
    x_dim: str,
    y_dim: str,
) -> xr.Dataset:
    """Normalize spatial coordinates to increasing x and decreasing y."""
    if ds[x_dim].values[0] > ds[x_dim].values[-1]:
        ds = ds.isel({x_dim: slice(None, None, -1)})

    if ds[y_dim].values[0] < ds[y_dim].values[-1]:
        ds = ds.isel({y_dim: slice(None, None, -1)})

    return ds
