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

import itertools
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import dask.array as da
import dask.array.core as dac
import numpy as np

IntTuple = tuple[int, ...]
SliceTuple = tuple[slice, ...]
IntIterable = Iterable[int]
IntTupleIterable = Iterable[IntTuple]
SliceTupleIterable = Iterable[SliceTuple]

_CLUSTER_TAGS_ENV_VAR_NAME = "XCUBE_DASK_CLUSTER_TAGS"
_CLUSTER_ACCOUNT_ENV_VAR_NAME = "XCUBE_DASK_CLUSTER_ACCOUNT"


def compute_array_from_func(
    func: Callable[..., np.ndarray],
    shape: IntTuple,
    chunks: IntTuple,
    dtype: Any,
    name: str = None,
    ctx_arg_names: Sequence[str] = None,
    args: Sequence[Any] = None,
    kwargs: Mapping[str, Any] = None,
) -> da.Array:
    """Compute a dask array using the provided user function
    *func*, *shape*, and chunking *chunks*.

    The user function is expected to output the array's data
    blocks using arguments specified by *ctx_arg_names*, *args*,
    and *kwargs* and is expected to return a numpy array.

    You can request array and current block context information
    by specifying the optional *ctx_arg_names* keyword argument
    that is a sequence of names of special arguments passed to
    *user_func*. The following are available:

    * ``shape``: The array's shape. A tuple of ints.
    * ``chunks``: The array's chunks. A tuple of tuple of ints.
    * ``dtype``: The array's numpy data type.
    * ``name``: The array's name. A string or ``None``.
    * ``block_id``: The block's unique ID. An integer number
        ranging from zero to number of blocks minus one.
    * ``block_index``: The block's index as a tuple of ints.
    * ``block_shape``: The block's shape as a tuple of ints.
    * ``block_slices``: The block's shape as a tuple of int pair tuples.

    Args:
        func: User function that is called for each block of the
            array using arguments specified by *ctx_arg_names*,
            *args*, and *kwargs*. It must return a numpy array of
            shape "block_shape" and type *dtype*.
        shape: The array's shape. A tuple of sizes for each
            array dimension.
        chunks: The array's chunking. A tuple of chunk sizes for
            each array dimension. Must be of same length as *shape*.
        dtype: The array's numpy data type.
        name: The array's name.
        ctx_arg_names: Sequence names of arguments that are passed
            before *args* and *kwargs* to the user function.
        args: Arguments passed to the user function.
        kwargs: Keyword-arguments passed to the user function.

    Returns: A chunked dask array.
    """
    ctx_arg_names = ctx_arg_names or []
    args = args or []
    kwargs = kwargs or {}

    chunk_sizes = tuple(get_chunk_sizes(shape, chunks))
    chunk_counts = tuple(get_chunk_counts(shape, chunks))
    block_indexes, block_shapes, block_slices = get_block_iterators(chunk_sizes)

    ctx_values = dict(
        shape=tuple(shape),
        chunks=chunk_sizes,
        dtype=dtype,
        name=name,
    )

    blocks = _NestedList(shape=chunk_counts)
    block_id = 0
    for chunk_index, chunk_shape, block_slices in zip(
        block_indexes, block_shapes, block_slices
    ):
        ctx_values.update(
            block_id=block_id,
            block_index=tuple(chunk_index),
            block_shape=tuple(chunk_shape),
            block_slices=tuple(
                (chunk_slice.start, chunk_slice.stop) for chunk_slice in block_slices
            ),
        )
        ctx_args = [ctx_values[ctx_arg_name] for ctx_arg_name in ctx_arg_names]
        block_id += 1

        # We use our own name here, because dac.from_func() tokenizes args which for
        # some reason takes forever
        block = dac.from_func(
            func,
            shape=chunk_shape,
            dtype=dtype,
            name=f"rectify_{name}-{uuid.uuid4()}",
            args=(*ctx_args, *args),
            kwargs=kwargs,
        )

        blocks[chunk_index] = block

    return da.block(blocks.data)


def get_chunk_sizes(shape: IntTuple, chunks: IntTuple) -> IntTupleIterable:
    for s, c in zip(shape, chunks):
        n = s // c
        if n * c < s:
            yield (c,) * n + (s % c,)
        else:
            yield (c,) * n


def get_chunk_counts(shape: IntTuple, chunks: IntTuple) -> Iterable[int]:
    for s, c in zip(shape, chunks):
        yield (s + c - 1) // c


def get_chunk_slice_tuples(chunk_size_tuples: IntTupleIterable) -> SliceTupleIterable:
    return (
        tuple(get_chunk_slices(chunk_size_tuple))
        for chunk_size_tuple in chunk_size_tuples
    )


def get_chunk_slices(chunk_sizes: Sequence[int]) -> Iterable[slice]:
    stop = 0
    for i in range(len(chunk_sizes)):
        start = stop
        stop = start + chunk_sizes[i]
        yield slice(start, stop)


def get_chunk_ranges(chunk_size_tuples: IntTupleIterable) -> Iterable[range]:
    return (range(len(chunk_size_tuple)) for chunk_size_tuple in chunk_size_tuples)


def get_block_iterators(
    chunk_sizes: IntTupleIterable,
) -> tuple[IntTupleIterable, IntTupleIterable, SliceTupleIterable]:
    chunk_sizes = tuple(chunk_sizes)
    chunk_slices_tuples = get_chunk_slice_tuples(chunk_sizes)
    chunk_ranges = get_chunk_ranges(chunk_sizes)
    block_indexes = itertools.product(*chunk_ranges)
    block_shapes = itertools.product(*chunk_sizes)
    block_slices = itertools.product(*chunk_slices_tuples)
    return block_indexes, block_shapes, block_slices


class _NestedList:
    """Utility class whose instances are used as input to dask.block()."""

    def __init__(self, shape: Sequence[int], fill_value: Any = None):
        self._shape = tuple(shape)
        self._data = self._new_data(shape, len(shape), fill_value, 0)

    @classmethod
    def _new_data(
        cls, shape: Sequence[int], ndim: int, fill_value: Any, dim: int
    ) -> list[list] | list[Any]:
        return [
            (
                cls._new_data(shape, ndim, fill_value, dim + 1)
                if dim < ndim - 1
                else fill_value
            )
            for _ in range(shape[dim])
        ]

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def data(self) -> list[list] | list[Any]:
        return self._data

    def __len__(self) -> int:
        return len(self._data)

    def __setitem__(self, index: int | slice | tuple, value: Any):
        data = self._data
        if isinstance(index, tuple):
            n = len(index)
            for i in range(n - 1):
                data = data[index[i]]
            data[index[n - 1]] = value
        else:
            data[index] = value

    def __getitem__(self, index: int | slice | tuple) -> Any:
        data = self._data
        if isinstance(index, tuple):
            n = len(index)
            for i in range(n - 1):
                data = data[index[i]]
            return data[index[n - 1]]
        else:
            return data[index]
