import unittest

from xcube_resampling.utils import _prep_interp_methods_downscale


class TestUtils(unittest.TestCase):

    def test__prep_interp_methods_downscale(self):
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
