from xcube_resampling.gridmapping import GridMapping
import numpy as np
import xarray as xr
import pyproj

GEO_CRS = pyproj.crs.CRS(4326)

gm = GridMapping.from_coords(
    x_coords=xr.DataArray(
        [
            [10.2, 10.3, 10.4, 10.5],
            [10.2, 10.3, 10.4, 10.5],
            [10.2, 10.3, 10.4, 10.5],
        ],
        dims=("lat", "lon"),
    ),
    y_coords=xr.DataArray(
        [
            [52.4, 52.4, 52.4, 52.4],
            [52.6, 52.6, 52.6, 52.6],
            [52.8, 52.8, 52.8, 52.8],
        ],
        dims=("lat", "lon"),
    ),
    crs=GEO_CRS,
)
