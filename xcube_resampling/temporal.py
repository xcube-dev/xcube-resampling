# The MIT License (MIT)
# Copyright (c) 2025 by the xcube development team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NON INFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd
import xarray as xr

from .constants import TemporalInterpMethods, TemporalAggMethods, LOG
from .utils import _select_variables


def resample_in_time(
    source_ds: xr.Dataset,
    frequency: str,
    *,
    variables: str | Iterable[str] | None = None,
    interp_methods: TemporalInterpMethods | None = None,
    agg_methods: TemporalAggMethods | None = None,
    offset=None,
    tolerance: float | Iterable[float] | str | None =None,
    metadata: dict[str, Any] = None,
) -> xr.Dataset:
    """Resample a dataset in the time dimension.

    *Important note:* As of xarray 0.14 and dask 2.8, the
    methods ``'median'`` and ``'percentile_<p>'` cannot be
    used if the variables in *cube* comprise chunked dask arrays.
    In this case, use the ``compute()`` or ``load()`` method
    to convert dask arrays into numpy arrays.

    Args:
        source_ds: The input xarray.Dataset. It should contain the `time`
            dimension.
        frequency: Temporal aggregation frequency. Use format
            "<count><offset>" where <offset> is one of 'H', 'D', 'W',
            'M', 'Q', 'Y'.
        variables: A single variable name or iterable of variable names to be
            resampled. If None, all data variables will be processed.
        interp_methods: Optional interpolation method to be used for
            upsampling the spatial variables in the temporal dimension. Can
            be a single interpolation method for all variables or a
            dictionary mapping variable names or dtypes to interpolation method.
            Supported methods include:

            - "linear",
            - "nearest",
            - "zero",
            - "slinear",
            - "quadratic",
            - "cubic",
            - "polynomial"

            The default is `linear`.
        agg_methods: Optional aggregation methods for downsampling spatial
            variables in the temporal dimension.
            Can be a single method for all variables, a list a methods for
            all variables or a dictionary mapping variable
            names or dtypes to method(s). Supported methods include:

            - "all",
            - "any",
            - "argmax",
            - "argmin",
            - "count",
            - "cumprod",
            - "cumsum",
            - "first",
            - "last",
            - "max",
            - "min",
            - "mean",
            - "median",
            - "percentile_<p>",
            - "std",
            - "sum",
            - "var"

            The default is `mean`.

             Note: The value ``'percentile_<p>'`` is a placeholder,
             where ``'<p>'`` must be replaced by an integer percentage
             value, e.g. ``'percentile_90'`` is the 90%-percentile.
        offset: Offset used to adjust the resampled time labels. Uses
            same syntax as *frequency*.
        tolerance: Time tolerance for selective upsampling methods.
            Defaults to *frequency*.
        metadata: Output metadata.

    Returns:
        A new xarray dataset resampled in time.
    """
    if frequency == "all":
        time_gap = np.array(source_ds.time[-1]) - np.array(source_ds.time[0])
        days = int((np.timedelta64(time_gap, "D") / np.timedelta64(1, "D")) + 1)
        frequency = f"{days}D"

    if variables:
        source_ds = _select_variables(source_ds, variables)

    guessed_operation = _analyze_resampling_operation(source_ds,
                                                      frequency,
                                                      interp_methods,
                                                      agg_methods
                                                      )

    resampled_cubes = []

    if guessed_operation == "agg":
        agg_methods_ = _prepare_methods(agg_methods, "mean")
        for var, methods in agg_methods_.items():
            resampled_cubes.extend(
                _apply_resampling(source_ds, var, methods, "agg", frequency,
                                  tolerance, offset)
            )

    elif guessed_operation == "interp":
        interp_methods_ = _prepare_methods(interp_methods, "linear")
        for var, methods in interp_methods_.items():
            resampled_cubes.extend(
                _apply_resampling(
                    source_ds, var, methods, "interp", frequency, tolerance, offset
                )
            )

    else:
        LOG.warning(
            "Could not determine resampling operation. Please pass agg_methods "
            "or interp_methods. Returning original dataset as is."
        )
        return source_ds

    if len(resampled_cubes) == 1:
        resampled_cube = resampled_cubes[0]
    else:
        resampled_cube = xr.merge(resampled_cubes)

    # TODO: add time_bnds to resampled_ds
    time_coverage_start = "%s" % source_ds.time[0]
    time_coverage_end = "%s" % source_ds.time[-1]

    resampled_cube.attrs.update(metadata or {})
    # TODO: add other time_coverage_ attributes
    resampled_cube.attrs.update(
        time_coverage_start=time_coverage_start, time_coverage_end=time_coverage_end
    )

    return resampled_cube

def _apply_resampling(ds, variable, methods, method_type, frequency, tolerance, offset):
    """Handles both agg and interp resampling."""
    results = []
    percentile_prefix = "percentile_"

    if variable == "all":
        source_ds = ds
    else:
        source_ds = ds[variable].to_dataset(name=variable)

    resampler = source_ds.resample(
        skipna=True, closed="left", label="left", time=frequency, offset=offset
    )

    for method in ([methods] if isinstance(methods, str) else methods):
        method_args, method_kwargs = [], {}
        method_postfix = method

        if method_type == "agg":
            if method.startswith(percentile_prefix):
                p = int(method[len(percentile_prefix):])
                method_args = [p / 100.0]
                method_postfix = f"p{p}"
                method = "quantile"
            method_kwargs = _get_agg_method_kwargs(method, frequency, tolerance)
        else:
            method_kwargs = _get_interp_method_kwargs(method)
            method = "interpolate"

        func = getattr(resampler, method)
        result = func(*method_args, **method_kwargs)

        result = result.rename({
            var: f"{var}_{method_postfix}"
            for var in result.data_vars
        })

        results.append(result)

    return results

def _prepare_methods(user_methods, default_method):
    if not user_methods:
        return {"all": default_method}
    if isinstance(user_methods, (str, list)):
        return {"all": user_methods}
    return user_methods

def _get_agg_method_kwargs(
        agg_method: str,
        frequency: str,
        tolerance: str,
        ):
    if agg_method in {
        "nearest",
        "bfill",
        "backfill",
        "ffill",
        "pad"
    }:
        kwargs = {"tolerance": tolerance or frequency}
    elif agg_method in {
        "first",
        "last",
        "sum",
        "cumsum",
        "cumprod",
        "min",
        "max",
        "mean",
        "median",
        "std",
        "var",
    }:
        kwargs = {"dim": "time", "keep_attrs": True, "skipna": True}
    elif agg_method == "prod":
        kwargs = {"dim": "time", "skipna": True, "keep_attrs": True}
    elif agg_method in {
        "all",
        "any"
        "count",
    }:
        kwargs = {"dim": "time", "keep_attrs": True}
    else:
        kwargs = {}
    return kwargs

def _get_interp_method_kwargs(
        interp_method: str,
        ):
    kwargs = {"kind": interp_method}
    return kwargs

def _analyze_resampling_operation(
    ds: xr.Dataset,
    frequency: str,
    interp_methods: TemporalInterpMethods | None = None,
    agg_methods: TemporalAggMethods | None = None,
) -> Literal["agg", "interp", None]:
    if "time" not in ds.dims:
        raise ValueError("Dataset must have a 'time' dimension.")

    if agg_methods and interp_methods:
        raise ValueError("Please provide either agg_methods or "
                         "interp_methods, not both.")
    if agg_methods:
        return "agg"

    if interp_methods:
        return "interp"

    time = ds["time"].values

    if len(time) < 2:
        raise ValueError("Not enough time points to resample.")

    tolerance = 0.05

    deltas = np.diff(time).astype("timedelta64[ns]").astype(float)
    mean_delta = np.mean(deltas)
    std_delta = np.std(deltas)

    if mean_delta == 0 or (std_delta / mean_delta) > tolerance:
        # irregular time delta in the time series
        return None

    mean_td = pd.to_timedelta(mean_delta, unit="ns")
    target_td = pd.Timedelta(frequency)

    ratio = mean_td / target_td

    if ratio < 1:
        return "agg"
    elif ratio > 1:
        return "interp"
    else:
        return None
