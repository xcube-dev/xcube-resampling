import unittest

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
    _prep_interp_methods_downscale,
    get_spatial_dims,
    clip_dataset_by_bbox,
    _select_variables,
    _get_grid_mapping_name,
    _get_interp_method,
    _get_agg_method,
    _get_recover_nan,
    _get_fill_value,
)


class TestUtils(unittest.TestCase):

    def test_get_spatial_dims_lon_lat(self):
        # Dataset with "lon" and "lat"
        ds = xr.Dataset(coords={"lon": [0, 1], "lat": [0, 1]})
        x_dim, y_dim = get_spatial_dims(ds)
        self.assertEqual((x_dim, y_dim), ("lon", "lat"))

    def test_get_spatial_dims_x_y(self):
        # Dataset with "x" and "y"
        ds = xr.Dataset(coords={"x": [0, 1], "y": [0, 1]})
        x_dim, y_dim = get_spatial_dims(ds)
        self.assertEqual((x_dim, y_dim), ("x", "y"))

    def test_get_spatial_dims_missing_dims(self):
        # Dataset with no recognized spatial dimensions
        ds = xr.Dataset(coords={"time": [0, 1]})
        with self.assertRaises(KeyError) as context:
            get_spatial_dims(ds)
        self.assertIn("No standard spatial dimensions found", str(context.exception))

    def test_clip_dataset_by_bbox_invalid_bbox(self):
        ds = xr.Dataset()
        with self.assertRaises(ValueError) as context:
            clip_dataset_by_bbox(ds, bbox=[0, 0, 1])
        self.assertIn("Expected bbox of length 4", str(context.exception))

    def test_clip_dataset_by_bbox(self):
        ds = xr.Dataset(
            {"data": (("lat", "lon"), [[1, 2], [3, 4]])},
            coords={"lon": [0, 1], "lat": [0, 1]},
        )
        clipped = clip_dataset_by_bbox(ds, bbox=[1, 1, 2, 2])
        self.assertTrue(clipped.sizes["lat"] == 1)
        self.assertTrue(clipped.sizes["lon"] == 1)

        bbox = [10, 10, 20, 20]
        with self.assertLogs("xcube.resampling", level="WARNING") as cm:
            _ = clip_dataset_by_bbox(ds, bbox=bbox)
        self.assertIn(
            "Clipped dataset contains at least one zero-sized dimension.", cm.output[0]
        )

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
