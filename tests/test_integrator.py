#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `Integrator` module."""


import unittest
from nose.tools import assert_equals, assert_raises

from pexmd import integrator
import numpy as np

class TestIntegrator(unittest.TestCase):
  """Tests for `Particles` module."""

  def setUp(self):
    """Set up test fixtures, if any."""
    self.x = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    self.v = np.array([[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
    self.a = np.array([[-100.0, 0.0, 0.0], [0.0, 200.0, 0.0]])

  def tearDown(self):
    """Tear down test fixtures, if any."""

  def test_create_integrator(self):
    """Create an empty object of generic particles."""
    integ = integrator.Integrator(0.1)

  def test_create_NVE(self):
    integ = integrator.NVE(0.1)

  def test_create_NVT(self):
    integ = integrator.NVT(0.1, 1.0)

  def test_first_step_integrator(self):
    integ = integrator.Integrator(0.1)
    x, v = integ.first_step(self.x, self.v, self.a)
    np.testing.assert_array_equal(x, self.x)
    np.testing.assert_array_equal(v, self.v)

  def test_last_step_integrator(self):
    integ = integrator.Integrator(0.1)
    x, v = integ.last_step(self.x, self.v, self.a)
    np.testing.assert_array_equal(x, self.x)
    np.testing.assert_array_equal(v, self.v)

  def test_first_step_velverlet(self):
    integ = integrator.VelVerlet(0.1)
    x, v = integ.first_step(self.x, self.v, self.a)
    np.testing.assert_array_almost_equal(x, self.x + 0.1*self.v + 0.5*self.a*0.01)
    np.testing.assert_array_almost_equal(v, self.v+0.5*0.1*self.a)

  def test_last_step_velverlet(self):
    integ = integrator.VelVerlet(0.1)
    x, v = integ.last_step(self.x, self.v, self.a)
    np.testing.assert_array_almost_equal(x, self.x)
    np.testing.assert_array_almost_equal(v, self.v+0.5*0.1*self.a)
