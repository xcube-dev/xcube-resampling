import unittest

import numpy as np
import xarray as xr

from xcube_resampling import mosaic_datasets

# noinspection PyProtectedMember
from xcube_resampling.mosaic import (
    _chunk_boundaries,
    _grid_index,
    _grid_size,
    _make_output_block,
    _mosaic_variable,
    _normalize_dataset,
    _validate_grid,
)


def _tile(
    x,
    y,
    values,
    *,
    variable="value",
    dtype=None,
    dims=("y", "x"),
):
    values = np.asarray(values, dtype=dtype)
    return xr.Dataset(
        {variable: (dims, values)},
        coords={"x": x, "y": y},
    )


class TestMosaicDatasets(unittest.TestCase):

    def test_single_tile_and_axis_normalization(self):
        ds = _tile(
            [2.0, 1.0],
            [0.0, 1.0],
            [[1, 2], [3, 4]],
        ).chunk(x=1, y=1)

        lazy = mosaic_datasets([ds], chunk_size=(1, 1))
        self.assertEqual(lazy.value.chunks, ((1, 1), (1, 1)))
        actual = lazy.compute()

        np.testing.assert_array_equal(actual.x, [1.0, 2.0])
        np.testing.assert_array_equal(actual.y, [1.0, 0.0])
        np.testing.assert_array_equal(actual.value, [[4, 3], [2, 1]])

    def test_chunk_size_defaults_to_first_dask_dataset(self):
        ds = _tile(
            [0.0, 1.0, 2.0],
            [2.0, 1.0, 0.0],
            np.arange(9).reshape(3, 3),
        ).chunk(x=2, y=2)

        actual = mosaic_datasets([ds])

        self.assertEqual(actual.value.chunks, ((2, 1), (2, 1)))

    def test_numpy_backed_inputs_return_computed_output(self):
        left = _tile([0.0, 1.0], [1.0, 0.0], [[1, 2], [3, 4]])
        right = _tile([2.0, 3.0], [1.0, 0.0], [[5, 6], [7, 8]])

        actual = mosaic_datasets([left, right])

        self.assertIsInstance(actual.value.data, np.ndarray)
        np.testing.assert_array_equal(
            actual.value,
            [[1, 2, 5, 6], [3, 4, 7, 8]],
        )

    def test_tiles_with_gap_are_filled(self):
        left = _tile([0.0, 1.0], [1.0, 0.0], [[1, 1], [1, 1]])
        right = _tile([4.0, 5.0], [1.0, 0.0], [[2, 2], [2, 2]])

        actual = mosaic_datasets([left, right], chunk_size=(2, 2)).compute()

        np.testing.assert_allclose(
            actual.value,
            [[1, 1, -1, -1, 2, 2], [1, 1, -1, -1, 2, 2]],
        )

    def test_overlapping_tiles_keep_first_non_fill_value(self):
        first = _tile([0.0, 1.0], [1.0, 0.0], [[1.0, np.nan], [3.0, 4.0]])
        second = _tile([1.0, 2.0], [1.0, 0.0], [[20.0, 30.0], [40.0, 50.0]])

        actual = mosaic_datasets([first, second], chunk_size=(3, 2)).compute()

        np.testing.assert_allclose(
            actual.value,
            [[1.0, 20.0, 30.0], [3.0, 4.0, 50.0]],
            equal_nan=True,
        )

    def test_integer_fill_value_replaces_fill_data(self):
        first = _tile([0.0, 1.0], [1.0, 0.0], [[1, 99], [3, 4]], dtype=np.int16)
        second = _tile([1.0, 2.0], [1.0, 0.0], [[20, 30], [40, 50]], dtype=np.int16)

        actual = mosaic_datasets(
            [first, second], fill_values=99, chunk_size=(3, 2)
        ).compute()

        np.testing.assert_array_equal(actual.value, [[1, 20, 30], [3, 4, 50]])

    def test_multidimensional_variable_and_custom_dimensions(self):
        data = np.arange(8).reshape(2, 2, 2)
        ds = xr.Dataset(
            {"value": (("time", "lat", "lon"), data)},
            coords={"time": [0, 1], "lon": [0.0, 1.0], "lat": [1.0, 0.0]},
        )

        actual = mosaic_datasets(
            [ds], x_dim="lon", y_dim="lat", chunk_size=(2, 2)
        ).compute()

        np.testing.assert_array_equal(actual.value, data)
        self.assertEqual(actual.value.dims, ("time", "lat", "lon"))

    def test_uint64_default_fill(self):
        ds = _tile([0.0, 1.0], [1.0, 0.0], [[1, 2], [3, 4]], dtype=np.uint64)
        other = _tile([3.0, 4.0], [1.0, 0.0], [[5, 6], [7, 8]], dtype=np.uint64)

        actual = mosaic_datasets([ds, other], chunk_size=(2, 2)).compute()

        self.assertEqual(actual.value.dtype, np.dtype(np.uint64))
        self.assertEqual(actual.value.values[0, 2], np.iinfo(np.uint64).max)

    def test_invalid_top_level_arguments(self):
        with self.assertRaisesRegex(ValueError, "At least one dataset"):
            mosaic_datasets([])

        ds = _tile([0.0, 1.0], [1.0, 0.0], [[1, 2], [3, 4]])
        with self.assertRaisesRegex(ValueError, "Chunk sizes must be positive"):
            mosaic_datasets([ds], chunk_size=(0, 2))

    def test_invalid_dataset_coordinates(self):
        ds = _tile([0.0, 1.0], [1.0, 0.0], [[1, 2], [3, 4]])

        missing = ds.drop_vars("x")
        with self.assertRaisesRegex(ValueError, "Dataset 0 must contain coordinates"):
            mosaic_datasets([missing])

        two_dimensional = ds.assign_coords(x=(("y", "x"), [[0.0, 1.0], [0.0, 1.0]]))
        with self.assertRaisesRegex(ValueError, "Only 1-D"):
            mosaic_datasets([two_dimensional])

        short = _tile([0.0], [1.0, 0.0], [[1], [2]])
        with self.assertRaisesRegex(ValueError, "at least two values"):
            mosaic_datasets([short])

    def test_invalid_later_dataset(self):
        first = _tile([0.0, 1.0], [1.0, 0.0], [[1, 2], [3, 4]])

        missing_coord = _tile([0.0, 1.0], [1.0, 0.0], [[5, 6], [7, 8]]).drop_vars("x")
        with self.assertRaisesRegex(ValueError, "Dataset 1 must contain coordinates"):
            mosaic_datasets([first, missing_coord])

        empty = xr.Dataset(
            {"value": (("y", "x"), np.empty((2, 0)))},
            coords={"x": [], "y": [1.0, 0.0]},
        )
        with self.assertRaisesRegex(ValueError, "dataset 1.*at least two values"):
            mosaic_datasets([first, empty])

        missing_var = xr.Dataset(
            {"other": (("y", "x"), np.ones((2, 2)))},
            coords={"x": [0.0, 1.0], "y": [1.0, 0.0]},
        )
        with self.assertRaisesRegex(ValueError, "missing data variables"):
            mosaic_datasets([first, missing_var])

    def test_incompatible_grids(self):
        first = _tile([0.0, 1.0], [1.0, 0.0], [[1, 2], [3, 4]])
        irregular = _tile([2.0, 3.5], [1.0, 0.0], [[5, 6], [7, 8]])
        with self.assertRaisesRegex(ValueError, "incompatible grid spacing"):
            mosaic_datasets([first, irregular])

        offset = _tile([0.5, 1.5], [1.0, 0.0], [[5, 6], [7, 8]])
        right = _tile([2.0, 3.0], [1.0, 0.0], [[9, 10], [11, 12]])
        with self.assertRaisesRegex(ValueError, "common grid"):
            mosaic_datasets([first, offset, right])

    def test_incompatible_non_spatial_dimension_size(self):
        first = xr.Dataset(
            {"value": (("time", "y", "x"), np.ones((2, 2, 2)))},
            coords={"time": [0, 1], "x": [0.0, 1.0], "y": [1.0, 0.0]},
        )
        second = xr.Dataset(
            {"value": (("time", "y", "x"), np.ones((1, 2, 2)))},
            coords={"time": [0], "x": [0.0, 1.0], "y": [1.0, 0.0]},
        )

        with self.assertRaisesRegex(ValueError, "incompatible size"):
            mosaic_datasets([first, second])


class TestMosaicHelpers(unittest.TestCase):

    def test_mosaic_variable_rejects_incompatible_dimensions(self):
        first = _tile([0.0, 1.0], [1.0, 0.0], [[1, 2], [3, 4]])

        class IncompatibleData:
            dims = ("y", "other_x")

            def transpose(self, *_dims):
                return self

        tile_specs = [
            {"dataset": first, "x_start": 0, "x_stop": 2, "y_start": 0, "y_stop": 2},
            {
                "dataset": {"value": IncompatibleData()},
                "x_start": 0,
                "x_stop": 2,
                "y_start": 0,
                "y_stop": 2,
            },
        ]

        with self.assertRaisesRegex(ValueError, "incompatible dimensions"):
            _mosaic_variable(
                tile_specs,
                name="value",
                x=np.array([0.0, 1.0]),
                y=np.array([1.0, 0.0]),
                x_dim="x",
                y_dim="y",
                fill_values=None,
                chunk_size=(2, 2),
            )

    def test_validate_grid(self):
        _validate_grid(np.array([0.0, 1.0, 2.0]), 1.0, "x")

        with self.assertRaisesRegex(ValueError, "at least two values"):
            _validate_grid(np.array([0.0]), 1.0, "x")
        with self.assertRaisesRegex(ValueError, "incompatible grid spacing"):
            _validate_grid(np.array([0.0, 1.5]), 1.0, "x")

    def test_grid_index(self):
        self.assertEqual(_grid_index(3.0, 1.0, 1.0, "x"), 2)
        self.assertEqual(_grid_index(1.0, 3.0, -1.0, "y"), 2)

        with self.assertRaisesRegex(ValueError, "common grid"):
            _grid_index(1.25, 0.0, 1.0, "x")

    def test_grid_size(self):
        self.assertEqual(_grid_size(0.0, 2.0, 1.0, "x"), 3)

        with self.assertRaisesRegex(ValueError, "Global 'x' extent"):
            _grid_size(0.0, 2.5, 1.0, "x")

    def test_chunk_boundaries(self):
        self.assertEqual(_chunk_boundaries(7, 3), [(0, 3), (3, 6), (6, 7)])
        self.assertEqual(_chunk_boundaries(0, 3), [])

    def test_normalize_dataset(self):
        ds = _tile([2.0, 1.0], [0.0, 1.0], [[1, 2], [3, 4]])
        normalized = _normalize_dataset(ds, x_dim="x", y_dim="y")
        np.testing.assert_array_equal(normalized.x, [1.0, 2.0])
        np.testing.assert_array_equal(normalized.y, [1.0, 0.0])

    def test_make_output_block_with_no_tiles(self):
        ref = _tile([0.0, 1.0], [1.0, 0.0], [[1.0, 2.0], [3.0, 4.0]])
        block = _make_output_block(
            [],
            name="value",
            y0=0,
            y1=2,
            x0=0,
            x1=3,
            fill_value=np.nan,
            y_dim="y",
            x_dim="x",
            ref_ds=ref,
        )
        np.testing.assert_allclose(
            block.compute(), np.full((2, 3), np.nan), equal_nan=True
        )
