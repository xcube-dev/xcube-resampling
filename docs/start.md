# Getting Started

The `xcube-resampling` package can be installed into an existing Python environment
using

```bash
pip install xcube-resampling
```

or

```bash
conda install -c conda-forge xcube-resampling
```

After installation, you are ready to go and use `resample_in_space` to resample you
datasets: 

### Generate a sample dataset

```python
import numpy as np
import xarray as xr

res = 0.1
source_ds = xr.Dataset(
    data_vars=dict(
        refl=xr.DataArray(
            np.array(
                [
                    [0, 1, 0, 2, 0, 3, 0, 4],
                    [2, 0, 3, 0, 4, 0, 1, 0],
                    [0, 4, 0, 1, 0, 2, 0, 3],
                    [1, 0, 2, 0, 3, 0, 4, 0],
                    [0, 3, 0, 4, 0, 1, 0, 2],
                    [4, 0, 1, 0, 2, 0, 3, 0],
                ],
                dtype=np.float64,
            ),
            dims=("lat", "lon"),
        )
    ),
    coords=dict(
        lon=xr.DataArray(50.0 + res * np.arange(0, 8) + 0.5 * res, dims="lon"),
        lat=xr.DataArray(10.6 - res * np.arange(0, 6) - 0.5 * res, dims="lat"),
    ),
)
```

### Apply resampling

```python
from xcube_resampling.spatial import resample_in_space
from xcube_resampling.gridmapping import GridMapping

target_gm = GridMapping.regular((3, 3), (50.05, 10.05), 0.2, "EPSG:4326")
target_ds = resample_in_space(source_ds, target_gm=target_gm)
```

```text
<xarray.Dataset> Size: 128B
Dimensions:      (lat: 3, lon: 3)
Coordinates:
    spatial_ref  int64 8B 0
  * lon          (lon) float64 24B 50.15 50.35 50.55
  * lat          (lat) float64 24B 10.55 10.35 10.15
Data variables:
    refl         (lat, lon) float64 72B 0.875 1.375 1.375 1.5 ... 1.25 1.5 1.0

```
