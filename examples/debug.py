import numpy as np
import xarray as xr

from xcube_resampling.gridmapping import GridMapping, CRS_WGS84
from xcube_resampling.rectify import rectify_dataset


def create_2x2_dataset_with_irregular_coords():
    lon = np.array([[1.0, 6.0], [0.0, 2.0]])
    lat = np.array([[56.0, 53.0], [52.0, 50.0]])
    rad = np.array([[1.0, 2.0], [3.0, 4.0]])
    return xr.Dataset(
        dict(
            lon=xr.DataArray(lon, dims=("y", "x")),
            lat=xr.DataArray(lat, dims=("y", "x")),
            rad=xr.DataArray(rad, dims=("y", "x")),
        )
    )


source_ds = create_2x2_dataset_with_irregular_coords()

target_gm = GridMapping.regular(size=(4, 4), xy_min=(-1, 49), xy_res=2, crs=CRS_WGS84)
target_ds = rectify_dataset(source_ds, target_gm=target_gm, spline_orders=0)
print(target_ds)
