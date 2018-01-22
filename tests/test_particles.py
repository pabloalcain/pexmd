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
    self.four_positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                                    [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    self.three_positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                                    [-1.0, 0.0, 0.0]])
    """Set up test fixtures, if any."""

  def tearDown(self):
    """Tear down test fixtures, if any."""

  def test_create_empty_base_particles(self):
    """Create an empty object of generic particles."""
    part = particles.Base()

  def test_create_empty_point_particles(self):
    part = particles.PointParticles()

  def test_add_particles_through_position(self):
    part = particles.Base()
    part.x = self.four_positions
    assert_equals(part.n, 4)
    sttr = lambda x: part.__setattr__("x", x)
    assert_raises(ValueError, sttr, self.three_positions)
