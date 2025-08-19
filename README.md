# xcube-resampling

[![Build Status](https://github.com/xcube-dev/xcube-resampling/actions/workflows/unit-tests.yml/badge.svg?branch=main)](https://github.com/xcube-dev/xcube-resampling/actions/workflows/unit-tests.yml)
[![codecov](https://codecov.io/gh/xcube-dev/xcube-resampling/graph/badge.svg?token=ktcp1maEgz)](https://codecov.io/gh/xcube-dev/xcube-resampling)
[![PyPI Version](https://img.shields.io/pypi/v/xcube-resampling)](https://pypi.org/project/xcube-resampling/)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/xcube-resampling/badges/version.svg)](https://anaconda.org/conda-forge/xcube-resampling)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/xcube-resampling/badges/license.svg)](https://anaconda.org/conda-forge/xcube-resampling)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**xcube-resampling** provides efficient algorithms for transforming datasets into 
different spatial grid mappings. It is designed for geospatial workflows that need 
flexible resampling and reprojection.

### ✨ Features
- **Affine resampling** – simple resampling using affine transformations  
- **Reprojection** – convert datasets between different coordinate reference systems (CRS)  
- **Rectification** – transform irregular grids into regular, well-structured grids  

All methods work seamlessly with chunked (lazily loaded) [xarray.Datasets](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html) and are powered by [Dask](https://www.dask.org/) for scalable, out-of-core computation.

### ⚡ Lightweight & Independent
The package is independent of the core *xcube* framework and has minimal dependencies:
`affine, dask, dask-image, numba, numpy, pyproj, xarray, zarr`.

Find out more in the [xcube-resampling Documentation](https://xcube-dev.github.io/xcube-resampling/).
