# -*- coding: utf-8 -*-

# The MIT License (MIT)
# Copyright (c) 2022 by the xcube development team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

__author__ = "Martin Böttcher, Brockmann Consult GmbH"
__copyright__ = "Copyright (c) 2022 by the xcube development team and contributors"
__license__ = "MIT"
__version__ = "1.0"
__email__ = "info@brockmann-consult.de"
__status__ = "Development"

# changes in 1.1
# ...

import numpy as np
import dask.array as da

class Interpolator:
    """
    Interpolates tie-point data to an associated image grid.
    The corners of the tie-point grid must match the corners of the
    image grid, and the step width of the tie points must be constant.
    Interpolation is bi-linear in pixel coordinates, not in geographic
    coordinates.

    The calling sequence is:

        sza_tp_data = sza_tp_image.compute()
        interpolator = Interpolator()
        sza_image = interpolator.tp_interpolate(sza_tp_data, oa01_radiance)

    """
    def tp_interpolate(self, tp_data, image_data, **kwargs):
        """
        Interpolates tie-point data to positions of image grid.
        :param tp_data: numpy (!) array with the tie-point data
        :param image_data: dask array, used as dummy to determine shape and chunksize
        :param kwargs: passed through to block function, empty
        :return: dask array with tp data interpolated to image grid
        """
        return da.map_blocks(Interpolator.tp_interpolate_block,
                             da.empty(shape=image_data.blocks.shape,
                                      chunks=(1, 1),
                                      dtype=np.byte),
                             tp_data=tp_data,
                             tp_step=((image_data.shape[0]-1) // (tp_data.shape[0]-1),
                                      (image_data.shape[1]-1) // (tp_data.shape[1]-1)),
                             image_shape=image_data.shape,
                             image_chunksize=image_data.chunksize,
                             meta=np.array([], dtype=tp_data.dtype),
                             **kwargs)

    @staticmethod
    def tp_interpolate_block(*inputs: np.ndarray, block_id=None, **kwargs):
        """
        Block-wise interpolation of tie-point data to the image grid
        using linear interpolation based on pixel coordinates (not geo-coordinates).
        TODO shall we support stacks of tiepoint data?
        :param inputs: ignored, a dummy 1*1 array is sufficient
        :param block_id: tuple of (block_row, block_col)
        :param kwargs: tp_data: numpy array, complete tp grid extent
                       tp_step: tuple of tie point step in y and x
                       image_shape: tuple of image shape
                       image_chunksize: tuple of image chunksize
        :return: numpy array of interpolated tp_data with the extent of the image block,
                 usually image_chunksize except for the right and lower border.
        """
        tp_data = kwargs["tp_data"]
        tp_step = kwargs["tp_step"]
        imagesize = kwargs["image_shape"]
        chunksize = kwargs["image_chunksize"]
        if (tp_data.shape[0] - 1) * tp_step[0] + 1 != imagesize[0] or \
           (tp_data.shape[1] - 1) * tp_step[1] + 1 != imagesize[1]:
            raise ValueError("tp grid does not match image grid: "
                             + str(tp_data.shape) + " "
                             + str(tp_step) + " "
                             + imagesize)

        # corner pixel coordinates of the block
        y1 = block_id[0]*chunksize[0]
        x1 = block_id[1]*chunksize[1]
        y2 = min(y1 + chunksize[0], imagesize[0])
        x2 = min(x1 + chunksize[1], imagesize[1])

        # extend tp grid by one column and one row for interpolation
        tp_height = tp_data.shape[0]
        tp_width = tp_data.shape[1]
        tp_y_step = tp_step[0]
        tp_x_step = tp_step[1]
        tp_dummy_column = np.zeros(tp_height).reshape((tp_height, 1))
        tp_dummy_row = np.zeros(tp_width+1)
        tp_data = np.vstack([np.hstack([tp_data, tp_dummy_column]), tp_dummy_row])

        # 1-D pixel coordinates, reference tp pixel coordinates, weights
        y = np.arange(y1, y2).reshape((y2-y1, 1))
        x = np.arange(x1, x2)
        y_tp = y // tp_y_step
        x_tp = x // tp_x_step
        wy = (y - y_tp * tp_y_step) / tp_y_step
        wx = (x - x_tp * tp_x_step) / tp_x_step

        # 2-D interpolation using numpy broadcasting
        result = (1-wy) * (1-wx) * tp_data[y_tp, x_tp] \
                 + (1-wy) * wx * tp_data[y_tp, x_tp+1] \
                 + wy * (1-wx) * tp_data[y_tp+1, x_tp] \
                 + wy * wx * tp_data[y_tp+1, x_tp+1]

        return result
