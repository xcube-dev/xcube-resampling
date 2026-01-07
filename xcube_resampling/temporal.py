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

from collections.abc import Hashable
from typing import Iterable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from .constants import (
    LOG,
    TemporalAggMethod,
    TemporalAggMethods,
    TemporalInterpMethod,
    TemporalInterpMethods,
)
from .utils import _select_variables


def resample_in_time(
    source_ds: xr.Dataset,
    frequency: str,
    *,
    variables: str | Iterable[str] | None = None,
    interp_methods: TemporalInterpMethods | None = None,
    agg_methods: TemporalAggMethods | None = None,
    offset: str | None = None,
    tolerance: str | None = None,
) -> xr.Dataset:
    """
    Resample a dataset along the time dimension.

    This function allows both **upsampling** (interpolation) and **downsampling**
    (aggregation) of temporal data variables. It wraps xarray's
    [`resample`](https://docs.xarray.dev/en/stable/generated/xarray.DataArray.resample.html)
    functionality and automatically determines whether to apply interpolation or
    aggregation based on the input frequency and dataset time resolution.

    Args:
        source_ds: Input xarray Dataset containing a `time` dimension.
        frequency: Target temporal frequency, following
            Pandas period aliases. Format `<count><period>`, where `<period>`
            may be one of 's', 'min', 'h', 'D', 'W', 'M', 'Q', 'Y'.
        variables: Optional. Names of variables to resample. If None, all
            data variables are processed, which have a time coordinate.
        interp_methods: Optional interpolation method(s) for upsampling. Can be:

            - A single method for all variables
            - A list of methods (applied sequentially, each output saved as
              `varname_method`)
            - A dictionary mapping variable names or dtypes to method(s)

            Available methods are defined in
            [`TemporalInterpMethod`](api.md/#xcube_resampling.constants.TemporalInterpMethod).
            Default is "nearest" for integer data, otherwise "linear".
        agg_methods: Optional aggregation method(s) for downsampling. Can be:

            - A single method for all variables
            - A list of methods (applied sequentially, each output saved as
              `varname_method`)
            - A dictionary mapping variable names or dtypes to method(s)

            Available methods are defined in
            [`TemporalAggMethod`](api.md/#xcube_resampling.constants.TemporalAggMethod).
            The placeholder `'percentile_<p>'` can be used for percentiles
            (e.g., `'percentile_90'`). Default is `'nearest'` for integer data,
            otherwise `'mean'`.
        offset: Optional offset to adjust resampled time labels. Uses the
            same syntax as frequency.
        tolerance: Optional maximum allowed distance for selective downsampling
            methods (e.g., `'backfill'`, `'ffill'`, `'nearest'`). Defaults to the
            resampling frequency.

    Returns:
        A new xarray Dataset resampled along the time dimension.

    Notes:
        - The function automatically decides whether to apply interpolation or
          aggregation based on the target frequency and the dataset’s temporal
          resolution, if `agg_methods` or `interp_methods` are not provided.
        - If the time series is highly irregular (coefficient of variation > 5%),
          explicit `agg_methods` or `interp_methods` must be provided.
    """

    if "time" not in source_ds.dims:
        raise ValueError("Dataset must have a 'time' dimension.")

    if variables:
        source_ds = _select_variables(source_ds, variables)

    if frequency == "all":
        days = int((source_ds.time[-1] - source_ds.time[0]) / np.timedelta64(1, "D"))
        frequency = f"{days + 1}D"

    guessed_operation = _guess_resampling_operation(
        source_ds, frequency, interp_methods, agg_methods
    )

    if guessed_operation == "agg":
        target_ds = _apply_aggregation(
            source_ds,
            frequency,
            agg_methods=agg_methods,
            offset=offset,
            tolerance=tolerance,
        )

    elif guessed_operation == "interp":
        target_ds = _apply_interpolation(
            source_ds,
            frequency,
            interp_methods=interp_methods,
            offset=offset,
        )

    else:
        LOG.warning(
            "Could not determine resampling operation. Please pass agg_methods "
            "or interp_methods. Returning original dataset as is."
        )
        return source_ds

    # TODO: add time_bnds to resampled_ds
    time_coverage_start = "%s" % source_ds.time[0]
    time_coverage_end = "%s" % source_ds.time[-1]

    target_ds.attrs.update(
        time_coverage_start=time_coverage_start,
        time_coverage_end=time_coverage_end,
    )

    return target_ds


def _apply_aggregation(
    dataset: xr.Dataset,
    frequency: str,
    agg_methods: TemporalAggMethods | None = None,
    offset: str | None = None,
    tolerance: str | None = None,
) -> xr.Dataset:
    percentile_prefix = "percentile_"

    resampled_dataset = xr.Dataset(attrs=dataset.attrs)
    for var_name, data_array in dataset.data_vars.items():
        if "time" not in data_array.coords:
            continue
        var_methods = _get_temporal_agg_method(agg_methods, var_name, data_array)
        resampler = data_array.resample(
            skipna=True, closed="left", label="left", time=frequency, offset=offset
        )
        for method in var_methods:
            method_args, method_kwargs = [], {}
            method_postfix = method
            if method.startswith(percentile_prefix):
                p = int(method[len(percentile_prefix) :])
                method_args = [p / 100.0]
                method_postfix = f"p{p}"
                method = "quantile"
            method_kwargs = _get_agg_method_kwargs(method, frequency, tolerance)
            func = getattr(resampler, method)
            if agg_methods is None or isinstance(agg_methods, str):
                var_name_out = var_name
            else:
                var_name_out = f"{var_name}_{method_postfix}"
            resampled_dataset[var_name_out] = func(*method_args, **method_kwargs)

    return resampled_dataset


def _apply_interpolation(
    dataset: xr.Dataset,
    frequency: str,
    interp_methods: TemporalInterpMethods | None = None,
    offset: str | None = None,
) -> xr.Dataset:
    resampled_dataset = xr.Dataset(attrs=dataset.attrs)
    for var_name, data_array in dataset.data_vars.items():
        if "time" not in data_array.coords:
            continue
        var_methods = _get_temporal_interp_method(interp_methods, var_name, data_array)
        resampler = data_array.resample(
            skipna=True, closed="left", label="left", time=frequency, offset=offset
        )
        for method in var_methods:
            func = getattr(resampler, "interpolate")
            if interp_methods is None or isinstance(interp_methods, str):
                var_name_out = var_name
            else:
                var_name_out = f"{var_name}_{method}"
            resampled_dataset[var_name_out] = func(kind=method)
    return resampled_dataset


def _get_agg_method_kwargs(
    agg_method: str,
    frequency: str,
    tolerance: str | None = None,
) -> dict:
    if agg_method in {"backfill", "bfill", "ffill", "nearest", "pad"}:
        kwargs = {"tolerance": tolerance or frequency}
    elif agg_method in {"all", "any", "count"}:
        kwargs = {"dim": "time", "keep_attrs": True}
    elif agg_method in {
        "cumprod",
        "cumsum",
        "first",
        "last",
        "max",
        "min",
        "mean",
        "median",
        "prod",
        "quantile",
        "std",
        "sum",
        "var",
    }:
        kwargs = {"dim": "time", "keep_attrs": True, "skipna": True}
    else:
        raise ValueError(f"Aggregation method {agg_method!r} not supported.")
    return kwargs


def _guess_resampling_operation(
    ds: xr.Dataset,
    frequency: str,
    interp_methods: TemporalInterpMethods | None = None,
    agg_methods: TemporalAggMethods | None = None,
) -> Literal["agg", "interp", None]:

    if agg_methods and interp_methods:
        raise ValueError(
            "Please provide either agg_methods or " "interp_methods, not both."
        )
    elif agg_methods:
        return "agg"
    elif interp_methods:
        return "interp"

    time = ds["time"].values
    if len(time) < 2:
        raise ValueError("Not enough time points to resample.")
    deltas = np.diff(time).astype("timedelta64[ns]").astype(float)
    mean_delta = np.mean(deltas)
    std_delta = np.std(deltas)
    if mean_delta == 0 or (std_delta / mean_delta) > 0.05:
        # irregular time delta in the time series
        return None

    mean_td = pd.to_timedelta(int(mean_delta), unit="ns")
    target_td = pd.Timedelta(frequency)
    ratio = mean_td / target_td
    if ratio < 1:
        return "agg"
    elif ratio > 1:
        return "interp"
    else:
        return None


def _get_temporal_interp_method(
    interp_methods: TemporalInterpMethods | None,
    key: Hashable,
    var: xr.DataArray,
) -> TemporalInterpMethod | Sequence[TemporalInterpMethod]:
    def assign_defaults(data_type: np.dtype) -> TemporalInterpMethod:
        return "nearest" if np.issubdtype(data_type, np.integer) else "linear"

    if isinstance(interp_methods, Mapping):
        interp_method = interp_methods.get(str(key), interp_methods.get(var.dtype))
        if interp_method is None:
            LOG.warning(
                f"Interpolation method could not be derived from the mapping "
                f"`interp_methods` for data variable {key!r} with data type "
                f"{var.dtype!r}. Defaults are assigned."
            )
            interp_method = assign_defaults(var.dtype)
    elif isinstance(interp_methods, str) or isinstance(interp_methods, Sequence):
        interp_method = interp_methods
    else:
        interp_method = assign_defaults(var.dtype)

    if isinstance(interp_method, str):
        interp_method = [interp_method]
    return interp_method


def _get_temporal_agg_method(
    agg_methods: TemporalAggMethods | None,
    key: Hashable,
    var: xr.DataArray,
) -> TemporalAggMethod | Sequence[TemporalAggMethod]:
    def assign_defaults(data_type: np.dtype) -> TemporalAggMethod:
        return "nearest" if np.issubdtype(data_type, np.integer) else "mean"

    if isinstance(agg_methods, Mapping):
        agg_method = agg_methods.get(str(key), agg_methods.get(var.dtype))
        if agg_method is None:
            LOG.warning(
                f"Aggregation method could not be derived from the mapping "
                f"`agg_methods` for data variable {key!r} with data type "
                f"{var.dtype!r}. Defaults are assigned."
            )
            agg_method = assign_defaults(var.dtype)
    elif isinstance(agg_methods, str) or isinstance(agg_methods, Sequence):
        agg_method = agg_methods
    else:
        agg_method = assign_defaults(var.dtype)

    if isinstance(agg_method, str):
        agg_method = [agg_method]
    return agg_method
