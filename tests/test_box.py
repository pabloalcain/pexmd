#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `Box` module."""


import unittest
from nose.tools import assert_equals, assert_raises

from pexmd import box
import numpy as np

class TestIntegrator(unittest.TestCase):
  """Tests for `Particles` module."""

  def setUp(self):
    """Set up test fixtures, if any."""
    self.x0 = np.array([-2.0, -1.0, 1.0])
    self.lx0 = [-2.0, -1.0, 1.0]
    self.xf = np.array([3.0, 4.0, 4.0])
    self.lxf = [3.0, 4.0, 4.0]
    self.x = np.array([[-1.0, 5.0, 2.0], [7.0, -3.0, -8.0]])
    self.v = np.array([[0.1, 0.1, -0.1], [-0.5, -0.3, -0.8]])

    self.x_pbc = np.array([[-1.0, 0.0, 2.0], [2.0, 2.0, 1.0]])
    self.v_pbc = np.array([[0.1, 0.1, -0.1], [-0.5, -0.3, -0.8]])

    self.x_fbc = np.array([[-1.0, 3.0, 2.0], [-1.0, 1.0, 4.0]])
    self.v_fbc = np.array([[0.1, -0.1, -0.1], [0.5, 0.3, 0.8]])

  def tearDown(self):
    """Tear down test fixtures, if any."""

  def test_create_box(self):
    """Create an empty periodic box."""
    b = box.Box(self.x0, self.xf, t='Periodic')

  def test_create_box_from_list(self):
    """Create an empty periodic box from a list."""
    b = box.Box(self.lx0, self.lxf, t='Periodic')
    np.testing.assert_array_almost_equal(b.x0, self.x0)
    np.testing.assert_array_almost_equal(b.xf, self.xf)

  def test_create_box_from_scalar(self):
    """Create an empty periodic box from an scalar."""
    b = box.Box(-1.5, 1.5, t='Periodic')
    np.testing.assert_array_almost_equal(b.x0, np.array([-1.5, -1.5, -1.5], dtype=np.float32))
    np.testing.assert_array_almost_equal(b.xf, np.array([1.5, 1.5, 1.5], dtype=np.float32))

  def test_wrap_periodic(self):
    """Wrap through PBC."""
    b = box.Box(self.x0, self.xf, t='Periodic')
    x, v = b.wrap_boundary(self.x, self.v)
    np.testing.assert_array_almost_equal(x, self.x_pbc)
    np.testing.assert_array_almost_equal(v, self.v_pbc)

  def test_wrap_fixed(self):
    """Wrap through FBC."""
    b = box.Box(self.x0, self.xf, t='Fixed')
    x, v = b.wrap_boundary(self.x, self.v)
    np.testing.assert_array_almost_equal(x, self.x_fbc)
    np.testing.assert_array_almost_equal(v, self.v_fbc)
