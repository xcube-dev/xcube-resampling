# The MIT License (MIT)
# Copyright (c) 2022 by the xcube development team and contributors
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import unittest
import pyproj
import numpy as np
import dask.array as da
from xcube_resampling.grid import Grid
from xcube_resampling.rectifier import Rectifier

# noinspection PyTypeChecker
class RectificationTest(unittest.TestCase):

    @classmethod
    def setUpClass(self) -> None:
        #self._client = Client(LocalCluster(n_workers=8, processes=True))
        #self._client = Client(n_workers=1, threads_per_worker=6)
        pass

    @classmethod
    def tearDownClass(self) -> None:
        #self._client.close()
        pass
    
    def test_forward_pixel_index(self):
        src_lon = np.array([[10.1, 10.3, 10.5, 10.7, 10.9, 11.1, 11.3, 11.5],
                          [10.0, 10.2, 10.4, 10.6, 10.8, 11.0, 11.2, 11.4],
                          [9.9, 10.1, 10.3, 10.5, 10.7, 10.9, 11.1, 11.3],
                          [9.8, 10.0, 10.2, 10.4, 10.6, 10.8, 11.0, 11.2],
                          [9.7, 9.9, 10.1, 10.3, 10.5, 10.7, 10.9, 11.1]])
        src_lat = np.array([[53.5, 53.48, 53.46, 53.44, 53.42, 53.4, 53.38, 53.36],
                          [53.3, 53.28, 53.26, 53.24, 53.22, 53.2, 53.18, 53.16],
                          [53.1, 53.08, 53.06, 52.04, 53.02, 53.0, 52.98, 52.96],
                          [52.9, 52.88, 52.86, 52.84, 52.82, 52.8, 52.78, 52.76],
                          [52.7, 52.68, 52.66, 52.64, 52.62, 52.6, 52.58, 52.56]])
        dst = Grid(pyproj.CRS(4326), (10.0, 53.4), (0.1, -0.1), (10, 7), (3, 3))
        r = Rectifier(self.src_lon, self.src_lat, dst_grid)
        index = r.create_forward_index()
        # TODO add assert
        print(index)
        assert index.shape[0] == 2

