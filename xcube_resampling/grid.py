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

class Axis:
    pass

class Axis:
    """
    Triple that defines a 1-D grid axis with start, step, and count.
    Origin is at the 'upper left' corner of the start pixel.
    """
    def __init__(self, start:np.float64,
                 step:np.float64,
                 count:int,
                 name:str=None):
        self.start = start
        self.step = step
        self.count = count
        self.name = name

    def __str__(self):
        if self.name:
            return "{} {},{},{}".format(self.name, self.start, self.step, self.count)
        else:
            return "{},{},{}".format(self.start, self.step, self.count)

    @staticmethod
    def from_coords(coords: np.ndarray,
                    name: str = None) -> Axis:
        step = (coords[-1] - coords[0]) / (coords.shape[0] - 1)
        return Axis(coords[0] - step / 2, step, coords.shape[0], name=name)

    def to_coords(self) -> np.ndarray:
        """returns a 1d array of coordinate values located at pixel centres"""
        half_step = 0.5 * self.step
        return np.linspace(self.start + half_step, self.start + self.count * self.step - half_step, self.count)

    
class Grid:
    """
    2-D grid area rows and columns, and with CRS that refers to locations on the Earth.
    Chunk sizes define blocks of data to be handled.
    Origin is the upper left corner of the start pixel.
    """
    def __init__(self, crs:pyproj.CRS,
                 rows: Axis, cols: Axis,
                 row_chunksize: int = None,
                 col_chunksize: int = None):
        self.crs = crs
        self.rows = rows
        self.cols = cols
        self.row_chunksize = row_chunksize
        self.col_chunksize = col_chunksize

    def __str__(self):
        return "rows=({}, {}, {}, {}), cols=({}, {}, {}, {}), crs={}".format(
            self.rows.start, self.rows.step, self.rows.count, self.row_chunksize,
            self.cols.start, self.cols.step, self.cols.count, self.col_chunksize,
            self.crs)

