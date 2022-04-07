# xcube-resampling

A Python package providing low-level, dask-aware spatial resampling algorithms 
for xcube and everyone. It collects implementations for

* reprojection
* rectification
* spatial interpolation
* spatial aggregation

between source and target grids. The grids may be

* projected with CRS, or pixel geo-coded
* on a coarser, finer, or almost equal resolution

The implementation makes use of dask, numpy, pyproj but comes with few
dependencies else. One layer of interfaces is on the level of dask arrays
and numpy arrays, to facilitate their integration into other software
packages that may not use xarray Dataset and DataArray.
