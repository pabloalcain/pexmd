#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `Particles` package."""


import unittest
from nose.tools import assert_equals, assert_raises

from pexmd import particles
import numpy as np

class TestParticles(unittest.TestCase):
  """Tests for `Particles` module."""

  def setUp(self):
    """Set up test fixtures, if any."""
    self.four_by3 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                              [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    self.three_by3 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                               [-1.0, 0.0, 0.0]])
    self.lfour_by3 = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                      [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    self.lthree_by3 = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                       [-1.0, 0.0, 0.0]]
  def tearDown(self):
    """Tear down test fixtures, if any."""

  def test_create_base_particles(self):
    """Create an empty object of generic particles."""
    part = particles.Base(3)

  def test_create_point_particles(self):
    part = particles.PointParticles(3)
    np.testing.assert_array_equal(part.idx, np.arange(3))

  def test_set_position_from_array(self):
    part = particles.PointParticles(4)
    part.x = self.four_by3
    assert_equals(part.n, 4)

  def test_set_position_from_list(self):
    part = particles.PointParticles(4)
    part.x = self.lfour_by3
    assert_equals(part.n, 4)
    assert_equals(type(part.x), np.ndarray)

  def test_set_position_wrong_size(self):
    part = particles.PointParticles(4)
    sttr = lambda x: part.__setattr__("x", x)
    assert_raises(ValueError, sttr, self.three_by3)


  def test_modify_position(self):
    part = particles.PointParticles(4)
    me = self.four_by3.copy()
    part.x = self.four_by3
    np.testing.assert_array_equal(part.x, self.four_by3)
    part.x[0, 0] = 1000
    np.testing.assert_array_equal(me, self.four_by3)

  def test_modify_velocity(self):
    part = particles.PointParticles(4)
    part.x = self.four_by3
    part.v = self.lfour_by3
    assert_equals(part.n, 4)
    sttr = lambda v: part.__setattr__("v", v)
    assert_raises(ValueError, sttr, self.three_by3)
    np.testing.assert_array_equal(part.v, self.four_by3)

  def test_modify_force(self):
    part = particles.PointParticles(4)
    part.f = self.four_by3
    assert_equals(part.n, 4)
    sttr = lambda f: part.__setattr__("f", f)
    assert_raises(ValueError, sttr, self.three_by3)
    np.testing.assert_array_equal(part.f, self.four_by3)

  def test_calculate_acceleration(self):
    part = particles.PointParticles(4)
    part.f = self.four_by3
    part.mass = 2.0
    np.testing.assert_array_equal(part.a, self.four_by3/2)

  def test_modify_type(self):
    part = particles.PointParticles(4)
    part.t = np.array([1, 1, 2, 2], dtype=np.int32)
    np.testing.assert_array_equal(part.t, np.array([1, 1, 2, 2]))
    part.t = 1
    np.testing.assert_array_equal(part.t, np.array([1, 1, 1, 1], dtype=np.int32))
    sttr = lambda t: part.__setattr__("t", t)
    assert_raises(ValueError, sttr, np.array([1, 1, 1]))

  def test_modify_mass(self):
    part = particles.PointParticles(4)
    part.mass = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)
    assert_equals(part.n, 4)
    part.mass = 1.0
    np.testing.assert_array_equal(part.mass, np.array([1, 1, 1, 1], dtype=np.float32))
    sttr = lambda mass: part.__setattr__("mass", mass)
    assert_raises(ValueError, sttr, np.array([1.0, 1.0, 1.0]))
