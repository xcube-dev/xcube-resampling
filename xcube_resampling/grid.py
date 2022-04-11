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
import pyproj
from typing import Tuple, Union, Optional
Number = Union[int, float]

class Grid:
    pass

class Grid:
    """
    2-D grid area with rows and columns, and with CRS that refers to locations on the Earth.
    The grid is represented by
    * crs: the coordinate reference system of the grid
    * xy_min: an origin, the upper left corner of the upmost and leftmost pixel
    * xy_res: step between pixels in the CRS unit, constant in each dimensino
    * width, height: number of columns and rows of the grid
    * tile_width, tile_height: number of columns and rows per block
    Origin is the upper left corner of the start pixel. But pixel x and y values are pixel centre coordinates.
    Tiles define blocks of data to be handled.
    """
    def __init__(self,
                 crs: pyproj.CRS,
                 xy_min: Tuple[Number, Number],
                 xy_res: Union[Number, Tuple[Number, Number]],
                 size: Union[int, Tuple[int, int]],
                 tile_size: Optional[Union[int, Tuple[int, int]]] = None):
        self.crs = crs
        # origin
        if xy_min is None:
            self.x_min, self.y_min = (None, None)
        else:
            self.x_min, self.y_min = xy_min
        # resolution
        if xy_res is None or isinstance(xy_res, (float, int)):
            self.x_res, self.y_res = (xy_res, xy_res)
        else:
            self.x_res, self.y_res = xy_res
        # extent
        if size is None or isinstance(size, int):
            self.width, self.height = (size, size)
        else:
            self.width, self.height = size
        # tile size
        if tile_size is None or isinstance(tile_size, int):
            self.tile_width, self.tile_height = (tile_size, tile_size)
        else:
            self.tile_width, self.tile_height = tile_size

    @staticmethod
    def from_coords(crs: pyproj.CRS,
                    axes: Tuple[np.ndarray, np.ndarray],
                    tile_size: Optional[Union[int, Tuple[int, int]]] = None) -> Grid:
        y_res = (axes[1][-1] - axes[1][0]) / (axes[1].shape[0] - 1)
        x_res = (axes[0][-1] - axes[0][0]) / (axes[0].shape[0] - 1)
        return Grid(crs,
                    (axes[0][0] - x_res / 2, axes[1][0] - y_res / 2),
                    (x_res, y_res),
                    (axes[0].shape[0], axes[1].shape[0]),
                    tile_size)

    def __str__(self):
        return "rows=({}, {}, {}, {}), cols=({}, {}, {}, {}), crs={}".format(
            self.y_min, self.y_res, self.height, self.tile_height,
            self.x_min, self.x_res, self.width, self.tile_width,
            self.crs)

    def y_axis(self) -> np.ndarray:
        """returns a 1d array of coordinate values located at pixel centres"""
        half_step = 0.5 * self.y_res
        return np.linspace(self.y_min + half_step, self.y_min + self.height * self.y_res - half_step, self.height)

    def x_axis(self) -> np.ndarray:
        """returns a 1d array of coordinate values located at pixel centres"""
        half_step = 0.5 * self.x_res
        return np.linspace(self.x_min + half_step, self.x_min + self.width * self.x_res - half_step, self.width)
