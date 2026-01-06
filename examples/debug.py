from xcube_resampling.gridmapping import GridMapping
from xcube_resampling import resample_in_space
import numpy as np
import xarray as xr
import pyproj

CRS_WGS84 = pyproj.crs.CRS(4326)


def create_4x4_dataset_with_irregular_coords():
    lon = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [0.0, 1.0, 2.0, 3.0],
            [-1.0, 0.0, 1.0, 2.0],
            [-2.0, -1.0, 0.0, 1.0],
        ]
    )
    lat = np.array(
        [
            [56.0, 55.0, 54.0, 53.0],
            [55.0, 54.0, 53.0, 52.0],
            [54.0, 53.0, 52.0, 51.0],
            [53.0, 52.0, 51.0, 50.0],
        ]
    )
    rad = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ]
    )
    return xr.Dataset(
        dict(
            rad=xr.DataArray(rad, dims=("y", "x")),
        ),
        coords=dict(
            lon=xr.DataArray(lon, dims=("y", "x")),
            lat=xr.DataArray(lat, dims=("y", "x")),
        ),
    )


source_ds = create_4x4_dataset_with_irregular_coords()
target_gm = GridMapping.regular(size=(2, 2), xy_min=(-1, 51), xy_res=2, crs=CRS_WGS84)
target_ds = resample_in_space(source_ds, target_gm=target_gm, interp_methods=1)
print(target_ds)
