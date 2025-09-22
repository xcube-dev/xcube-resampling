import unittest

import dask.array as da
import numpy as np
import xarray as xr

from xcube_resampling.constants import (
    AGG_METHODS,
    FILLVALUE_INT,
    FILLVALUE_UINT8,
    FILLVALUE_UINT16,
)

# noinspection PyProtectedMember
from xcube_resampling.utils import (
    _get_agg_method,
    _get_fill_value,
    _get_grid_mapping_name,
    _get_interp_method,
    _get_recover_nan,
    _prep_interp_methods_downscale,
    _select_variables,
    bbox_overlap,
    clip_dataset_by_bbox,
    get_spatial_coords,
    reproject_bbox,
)


class TestUtils(unittest.TestCase):

    def test_get_spatial_coords_lon_lat(self):
        # Dataset with "lon" and "lat"
        ds = xr.Dataset(coords={"lon": [0, 1], "lat": [0, 1]})
        x_dim, y_dim = get_spatial_coords(ds)
        self.assertEqual((x_dim, y_dim), ("lon", "lat"))

    def test_get_spatial_coords_longitude_latitude(self):
        # Dataset with "longitude" and "latitude"
        ds = xr.Dataset(coords={"longitude": [0, 1], "latitude": [0, 1]})
        x_dim, y_dim = get_spatial_coords(ds)
        self.assertEqual((x_dim, y_dim), ("longitude", "latitude"))

    def test_get_spatial_coords_x_y(self):
        # Dataset with "x" and "y"
        ds = xr.Dataset(coords={"x": [0, 1], "y": [0, 1]})
        x_dim, y_dim = get_spatial_coords(ds)
        self.assertEqual((x_dim, y_dim), ("x", "y"))

    def test_get_spatial_coords_missing_dims(self):
        # Dataset with no recognized spatial dimensions
        ds = xr.Dataset(coords={"time": [0, 1]})
        with self.assertRaises(KeyError) as context:
            get_spatial_coords(ds)
        self.assertIn("No standard spatial coordinates found", str(context.exception))

    def test_select_variables(self):
        ds = xr.Dataset(
            {
                "var1": ("x", [1, 2, 3]),
                "var2": ("x", [4, 5, 6]),
                "var3": ("x", [7, 8, 9]),
            },
            coords={"x": [0, 1, 2]},
        )

        # if variables=None, return dataset with all data variables
        result = _select_variables(ds, variables=None)
        self.assertEqual(set(result.data_vars), set(ds.data_vars))

        # select one variable
        result = _select_variables(ds, variables="var1")
        self.assertEqual(list(result.data_vars), ["var1"])
        self.assertTrue("var1" in result)

        # select multiple variables
        result = _select_variables(ds, variables=["var1", "var3"])
        self.assertEqual(set(result.data_vars), {"var1", "var3"})
        self.assertTrue("var2" not in result)

        # selecting a variable not in dataset should raise KeyError
        with self.assertRaises(KeyError):
            _select_variables(ds, variables="nonexistent_var")

    def test_n_get_grid_mapping_name(self):
        # no grid mapping
        ds = xr.Dataset({"var1": ("x", [1, 2, 3])}, coords={"x": [0, 1, 2]})
        self.assertIsNone(_get_grid_mapping_name(ds))

        # grid mapping in variables attribute
        ds = xr.Dataset({"var1": ("x", [1, 2, 3])})
        ds["var1"].attrs["grid_mapping"] = "crs_var"
        self.assertEqual(_get_grid_mapping_name(ds), "crs_var")

        # grid mapping in crs variable
        ds = xr.Dataset({"var1": ("x", [1, 2, 3]), "crs": 0}, coords={"x": [0, 1, 2]})
        self.assertEqual(_get_grid_mapping_name(ds), "crs")

        # grid mapping in spatial ref coordinate
        ds = xr.Dataset(
            {"var1": ("x", [1, 2, 3])}, coords={"x": [0, 1, 2], "spatial_ref": 0}
        )
        self.assertEqual(_get_grid_mapping_name(ds), "spatial_ref")

        # if multiple grid mapping found, error should be raised.
        ds = xr.Dataset({"var1": ("x", [1, 2, 3])})
        ds["var1"].attrs["grid_mapping"] = "gm1"
        ds["crs"] = 0
        with self.assertRaises(AssertionError):
            _get_grid_mapping_name(ds)

    def test_get_interp_method(self):
        int_var = xr.DataArray(np.array([1, 2, 3], dtype=np.int32), dims=["x"])
        float_var = xr.DataArray(
            np.array([1.0, 2.0, 3.0], dtype=np.float32), dims=["x"]
        )

        # integer type data array
        result = _get_interp_method(None, "var", int_var)
        self.assertEqual(result, 0)

        # float type data array
        result = _get_interp_method(None, "var", float_var)
        self.assertEqual(result, 1)

        # integer scalar
        result = _get_interp_method(1, "var", float_var)
        self.assertEqual(result, 1)

        # string
        result = _get_interp_method("nearest", "var", int_var)
        self.assertEqual(result, "nearest")

        # key matching
        interp_methods = {"var": "bilinear"}
        # noinspection PyTypeChecker
        result = _get_interp_method(interp_methods, "var", float_var)
        self.assertEqual(result, "bilinear")

        # dtaa type matching
        interp_methods = {np.dtype("float32"): "bilinear"}
        # noinspection PyTypeChecker
        result = _get_interp_method(interp_methods, "other", float_var)
        self.assertEqual(result, "bilinear")

        # no matching keys shall trigger a log warning
        interp_methods = {"something": "bilinear"}
        with self.assertLogs("xcube.resampling", level="WARNING") as cm:
            # noinspection PyTypeChecker
            result = _get_interp_method(interp_methods, "var", int_var)
        self.assertEqual(result, 0)  # default value
        self.assertIn("Defaults are assigned", cm.output[0])

    def test_prep_interp_methods_downscale(self):
        self.assertIsNone(_prep_interp_methods_downscale(None))
        self.assertEqual(_prep_interp_methods_downscale("triangular"), "bilinear")
        self.assertEqual(_prep_interp_methods_downscale("nearest"), "nearest")
        self.assertEqual(_prep_interp_methods_downscale(1), 1)

        interp_map = {"a": "triangular", "b": "nearest"}
        expected = {"a": "bilinear", "b": "nearest"}
        # noinspection PyTypeChecker
        self.assertEqual(_prep_interp_methods_downscale(interp_map), expected)

        interp_map = {"a": "nearest", "b": "bilinear"}
        # noinspection PyTypeChecker
        self.assertEqual(_prep_interp_methods_downscale(interp_map), interp_map)

    def test_get_agg_method(self):
        int_var = xr.DataArray(np.array([1, 2, 3], dtype=np.int32), dims=["x"])
        float_var = xr.DataArray(
            np.array([1.0, 2.0, 3.0], dtype=np.float32), dims=["x"]
        )

        # integer type data array, default
        result = _get_agg_method(None, "var", int_var)
        self.assertEqual(result, AGG_METHODS["center"])

        # float type data array, default
        result = _get_agg_method(None, "var", float_var)
        self.assertEqual(result, AGG_METHODS["mean"])

        # string as method
        result = _get_agg_method("center", "var", float_var)
        self.assertEqual(result, AGG_METHODS["center"])

        # key matching
        agg_methods = {"var": "mean"}
        # noinspection PyTypeChecker
        result = _get_agg_method(agg_methods, "var", int_var)
        self.assertEqual(result, AGG_METHODS["mean"])

        # data type matching
        agg_methods = {np.dtype("float32"): "mean"}
        # noinspection PyTypeChecker
        result = _get_agg_method(agg_methods, "other", float_var)
        self.assertEqual(result, AGG_METHODS["mean"])

        # no matching keys triggers log warning
        agg_methods = {"something": "mean"}
        with self.assertLogs("xcube.resampling", level="WARNING") as cm:
            # noinspection PyTypeChecker
            result = _get_agg_method(agg_methods, "var", int_var)
        self.assertEqual(result, AGG_METHODS["center"])  # default value
        self.assertIn("Defaults are assigned", cm.output[0])

    def test_get_recover_nan(self):
        int_var = xr.DataArray(np.array([1, 2, 3], dtype=np.int32), dims=["x"])
        float_var = xr.DataArray(
            np.array([1.0, 2.0, 3.0], dtype=np.float32), dims=["x"]
        )

        # bool directly
        result = _get_recover_nan(True, "var", int_var)
        self.assertTrue(result)

        result = _get_recover_nan(False, "var", float_var)
        self.assertFalse(result)

        # key mapping
        recover_nans = {"var": True}
        # noinspection PyTypeChecker
        result = _get_recover_nan(recover_nans, "var", int_var)
        self.assertTrue(result)

        # dtype mapping
        recover_nans = {np.dtype("float32"): True}
        # noinspection PyTypeChecker
        result = _get_recover_nan(recover_nans, "other", float_var)
        self.assertTrue(result)

        # missing key/dtype → default False with log warning
        recover_nans = {"something": True}
        with self.assertLogs("xcube.resampling", level="WARNING") as cm:
            # noinspection PyTypeChecker
            result = _get_recover_nan(recover_nans, "var", int_var)
        self.assertFalse(result)
        self.assertIn("Defaults are assigned", cm.output[0])

        # recover_nans is None → fallback default False
        result = _get_recover_nan(None, "var", float_var)
        self.assertFalse(result)

    def test_get_fill_value(self):
        uint8_var = xr.DataArray(np.array([1, 2, 3], dtype=np.uint8), dims=["x"])
        uint16_var = xr.DataArray(np.array([1, 2, 3], dtype=np.uint16), dims=["x"])
        int_var = xr.DataArray(np.array([1, 2, 3], dtype=np.int32), dims=["x"])
        float_var = xr.DataArray(
            np.array([1.0, 2.0, 3.0], dtype=np.float32), dims=["x"]
        )

        # scalar int
        result = _get_fill_value(-99, "var", int_var)
        self.assertEqual(result, -99)

        # scalar float
        result = _get_fill_value(-9.9, "var", float_var)
        self.assertEqual(result, -9.9)

        # mapping by variable name
        result = _get_fill_value({"var": 1234}, "var", int_var)
        self.assertEqual(result, 1234)

        # mapping by dtype
        result = _get_fill_value({np.dtype("float32"): 3.14}, "other", float_var)
        self.assertEqual(result, 3.14)

        # unmatched mapping triggers warning + defaults
        with self.assertLogs("xcube.resampling", level="WARNING") as cm:
            result = _get_fill_value({"something": 42}, "var", int_var)
        self.assertEqual(result, FILLVALUE_INT)
        self.assertIn("Fill value could not be derived", cm.output[0])

        # defaults
        self.assertEqual(_get_fill_value(None, "var", uint8_var), FILLVALUE_UINT8)
        self.assertEqual(_get_fill_value(None, "var", uint16_var), FILLVALUE_UINT16)
        self.assertEqual(_get_fill_value(None, "var", int_var), FILLVALUE_INT)
        self.assertTrue(np.isnan(_get_fill_value(None, "var", float_var)))

    def test_reproject_bbox(self):
        bbox_wgs84 = [2, 50, 3, 51]
        crs_wgs84 = "EPSG:4326"
        crs_3035 = "EPSG:3035"
        bbox_3035 = [3748675.9529771, 3011432.8944597, 3830472.1359979, 3129432.4914285]
        self.assertEqual(bbox_wgs84, reproject_bbox(bbox_wgs84, crs_wgs84, crs_wgs84))
        self.assertEqual(bbox_3035, reproject_bbox(bbox_3035, crs_3035, crs_3035))
        np.testing.assert_almost_equal(
            reproject_bbox(bbox_wgs84, crs_wgs84, crs_3035), bbox_3035
        )
        np.testing.assert_almost_equal(
            reproject_bbox(
                reproject_bbox(bbox_wgs84, crs_wgs84, crs_3035), crs_3035, crs_wgs84
            ),
            [
                1.829619451017442,
                49.93464594063249,
                3.1462425554926226,
                51.06428203128216,
            ],
        )

    def test_bbox_overlap(self):
        # identical boxes
        bbox = (0, 0, 10, 10)
        self.assertEqual(bbox_overlap(bbox, bbox), 1.0)

        # partial overlap
        source = (0, 0, 10, 10)
        target = (5, 5, 15, 15)
        # overlap area = 25, source area = 100
        expected = 25 / 100
        self.assertAlmostEqual(bbox_overlap(source, target), expected)

        # no overlap
        source = (0, 0, 10, 10)
        target = (20, 20, 30, 30)
        self.assertAlmostEqual(bbox_overlap(source, target), 0.0)

        # target fully inside source
        source = (0, 0, 10, 10)
        target = (2, 2, 8, 8)
        # overlap area = 36, source area = 100
        expected = 0.36
        self.assertAlmostEqual(bbox_overlap(source, target), expected)

        # source fully inside target
        source = (2, 2, 8, 8)
        target = (0, 0, 10, 10)
        # overlap area = source area = 36
        expected = 36 / 36
        self.assertAlmostEqual(bbox_overlap(source, target), expected)


class TestClipDatasetByBBox(unittest.TestCase):

    def setUp(self):
        # 1D coordinates dataset
        x = np.linspace(1, 10, 10)
        y = np.linspace(1, 20, 20)
        data_1d = np.random.rand(len(y), len(x))
        self.ds_1d = xr.Dataset({"var": (("y", "x"), data_1d)}, coords={"x": x, "y": y})

        # 2D coordinates dataset
        x2d, y2d = np.meshgrid(x, y)
        data_2d = np.random.rand(*x2d.shape)
        self.ds_2d = xr.Dataset(
            {"var": (("y", "x"), data_2d)},
            coords={"lon": (("y", "x"), x2d), "lat": (("y", "x"), y2d)},
        )

        # 2D coordinates dataset as dark array
        x2d, y2d = np.meshgrid(x, y)
        data_2d = np.random.rand(*x2d.shape)
        self.ds_2d_chunked = xr.Dataset(
            {"var": (("row", "column"), data_2d)},
            coords={
                "lon": (("row", "column"), da.from_array(x2d, chunks=(5, 5))),
                "lat": (("row", "column"), da.from_array(y2d, chunks=(5, 5))),
            },
        )

    def test_clip_1dcoord_inside_bbox(self):
        bbox = [2, 5, 8, 15]  # xmin, ymin, xmax, ymax
        clipped = clip_dataset_by_bbox(self.ds_1d, bbox, spatial_coords=("x", "y"))
        self.assertTrue((clipped.x >= bbox[0]).all())
        self.assertTrue((clipped.x <= bbox[2]).all())
        self.assertTrue((clipped.y >= bbox[1]).all())
        self.assertTrue((clipped.y <= bbox[3]).all())

    def test_clip_2dcoord_inside_bbox(self):
        bbox = [2, 5, 8, 15]
        clipped = clip_dataset_by_bbox(self.ds_2d, bbox)
        self.assertTrue((clipped["lon"].values >= bbox[0]).all())
        self.assertTrue((clipped["lon"].values <= bbox[2]).all())
        self.assertTrue((clipped["lat"].values >= bbox[1]).all())
        self.assertTrue((clipped["lat"].values <= bbox[3]).all())

    def test_clip_2dcoord_inside_bbox_chunked(self):
        bbox = [2, 5, 8, 15]
        clipped = clip_dataset_by_bbox(self.ds_2d_chunked, bbox)
        self.assertTrue((clipped["lon"].values >= bbox[0]).all())
        self.assertTrue((clipped["lon"].values <= bbox[2]).all())
        self.assertTrue((clipped["lat"].values >= bbox[1]).all())
        self.assertTrue((clipped["lat"].values <= bbox[3]).all())

    def test_clip_dataset_by_bbox_invalid_bbox(self):
        with self.assertRaises(ValueError) as context:
            clip_dataset_by_bbox(self.ds_1d, bbox=[0, 0, 1])
        self.assertIn("Expected bbox of length 4", str(context.exception))

    def test_unsupported_coord_dims(self):
        ds = self.ds_1d.copy()
        ds["x"] = ds["x"].expand_dims("z")  # 2D+ coordinate
        with self.assertRaises(ValueError):
            clip_dataset_by_bbox(ds, [0, 0, 5, 5])

    def test_bbox_outside_1d_dataset(self):
        bbox = [100, 100, 110, 110]  # completely outside
        with self.assertLogs("xcube.resampling", level="WARNING") as cm:
            clipped = clip_dataset_by_bbox(self.ds_1d, bbox)
        self.assertIn(
            "Clipped dataset contains at least one zero-sized dimension.", cm.output[0]
        )
        # should result in zero-sized dimensions
        self.assertTrue(any(size == 0 for size in clipped.sizes.values()))

    def test_bbox_outside_2d_dataset(self):
        bbox = [100, 100, 110, 110]  # completely outside
        with self.assertLogs("xcube.resampling", level="WARNING") as cm:
            clipped = clip_dataset_by_bbox(self.ds_2d, bbox)
        self.assertIn(
            "Clipped dataset contains at least one zero-sized dimension.", cm.output[0]
        )
        # should result in zero-sized dimensions
        self.assertTrue(any(size == 0 for size in clipped.sizes.values()))
