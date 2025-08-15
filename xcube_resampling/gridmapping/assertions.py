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

from collections.abc import Container
from typing import Any

_DEFAULT_NAME = "value"


def assert_given(
    value: Any, name: str = None, exception_type: type[Exception] = ValueError
):
    """Assert *value* is not False when converted into a Boolean value.
    Otherwise, raise *exception_type*.

    Args:
        value: The value to test.
        name: Name of a variable that holds *value*.
        exception_type: The exception type. Default is ``ValueError``.
    """
    if not value:
        raise exception_type(f"{name or _DEFAULT_NAME} must be given")


def assert_instance(
    value: Any,
    dtype: type | tuple[type, ...],
    name: str | None = None,
    exception_type: type[Exception] = TypeError,
):
    """Assert *value* is an instance of data type *dtype*.
    Otherwise, raise *exception_type*.

    Args:
        value: The value to test.
        dtype: A type or tuple of types.
        name: Name of a variable that holds *value*.
        exception_type: The exception type. Default is ``TypeError``.
    """
    if not isinstance(value, dtype):
        raise exception_type(
            f"{name or _DEFAULT_NAME} must be an instance of {dtype}, was {type(value)}"
        )


def assert_in(
    value: Any,
    container: Container,
    name: str = None,
    exception_type: type[Exception] = ValueError,
):
    """Assert *value* is a member of *container*.
    Otherwise, raise *exception_type*.

    Args:
        value: The value to test for membership.
        container: The container.
        name: Name of a variable that holds *value*.
        exception_type: The exception type. Default is ``ValueError``.
    """
    if value not in container:
        raise exception_type(f"{name or _DEFAULT_NAME} must be one of {container}")


def assert_true(value: Any, message: str, exception_type: type[Exception] = ValueError):
    """Assert *value* is true after conversion into a Boolean value.
    Otherwise, raise *exception_type*.

    Args:
        value: The value to test.
        message: The error message used if the assertion fails.
        exception_type: The exception type. Default is ``ValueError``.
    """
    if not value:
        raise exception_type(message)
