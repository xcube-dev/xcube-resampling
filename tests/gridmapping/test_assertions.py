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

import unittest

from xcube_resampling.gridmapping import assertions


class Testassertions(unittest.TestCase):
    # --- assert_given ---
    def test_assert_given_ok(self):
        assertions.assert_given("hello")

    def test_assert_given_fail(self):
        with self.assertRaises(ValueError) as cm:
            assertions.assert_given("", name="arg")
        self.assertIn("arg must be given", str(cm.exception))

    # --- assert_instance ---
    def test_assert_instance_ok(self):
        assertions.assert_instance(5, int)

    def test_assert_instance_fail(self):
        with self.assertRaises(TypeError) as cm:
            assertions.assert_instance("s", int, name="val")
        self.assertIn("val must be an instance of", str(cm.exception))

    def test_assert_instance_tuple_dtype(self):
        assertions.assert_instance(5, (int, float))  # should pass

    # --- assert_in ---
    def test_assert_in_ok(self):
        assertions.assert_in(1, [1, 2, 3])

    def test_assert_in_fail(self):
        with self.assertRaises(ValueError) as cm:
            assertions.assert_in("z", ["a", "b"], name="char")
        self.assertIn("char must be one of", str(cm.exception))

    # --- assert_true ---
    def test_assert_true_ok(self):
        assertions.assert_true(True, "must be true")

    def test_assert_true_fail(self):
        with self.assertRaises(ValueError) as cm:
            assertions.assert_true(False, "bad value")
        self.assertEqual(str(cm.exception), "bad value")
