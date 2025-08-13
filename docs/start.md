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

