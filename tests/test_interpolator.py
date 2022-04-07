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
import numpy as np
import dask.array as da
from xcube_resampling.interpolator import Interpolator

# noinspection PyTypeChecker
class InterpolationTest(unittest.TestCase):

    @classmethod
    def setUpClass(self) -> None:
        #self._client = Client(LocalCluster(n_workers=8, processes=True))
        #self._client = Client(n_workers=1, threads_per_worker=6)
        pass

    @classmethod
    def tearDownClass(self) -> None:
        #self._client.close()
        pass

    def test_interpolate_block(self):
        dummy_array = np.zeros((1,1), dtype=np.int8)
        imagesize = (9, 16)  # 9 rows, 16 columns
        chunksize = (3, 4)  # 3 row blocks a 3 rows, 4 col blocks a 4 rows
        tp_step = (2, 5)  # 5 rows, 4 columns
        tp_data = np.array([[0, 5, 10, 15],
                            [32, 37, 42, 47],
                            [64, 69, 74, 79],
                            [96, 101, 106, 111],
                            [128, 133, 138, 143]])
        block_id = (2, 1)  # 3rd row, 2nd column
        result = Interpolator.tp_interpolate_block(dummy_array,
                                                   block_id=block_id,
                                                   tp_data=tp_data,
                                                   tp_step=tp_step,
                                                   image_shape=imagesize,
                                                   image_chunksize=chunksize)
        assert result.shape == chunksize
        assert result[0, 0] == 100
        assert result[0, -1] == 103
        assert result[-1, 0] == 132
        assert result[-1, -1] == 135
        print(result)

    def test_interpolate(self):
        image = da.arange(9 * 16).reshape((9, 16)).rechunk((3, 4))
        tp_data = np.array([[0, 5, 10, 15],
                            [32, 37, 42, 47],
                            [64, 69, 74, 79],
                            [96, 101, 106, 111],
                            [128, 133, 138, 143]])
        interpolator = Interpolator()
        dask_array = interpolator.tp_interpolate(tp_data, image)
        dask_array.blocks.shape == (3, 4)
        result = dask_array.compute()
        assert result[6, 4] == 100
        assert result[6, 7] == 103
        assert result[8, 4] == 132
        assert result[8, 7] == 135
        print(result)
