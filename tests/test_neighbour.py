#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `Interaction` module."""


import unittest
from nose.tools import assert_equals
from pexmd import neighbour
import numpy as np

class TestNeighbour(unittest.TestCase):
  """Tests for `Neighbour` module."""

  def setUp(self):
    """Set up test fixtures, if any."""
    self.four_types = np.array([1, 1, 1, 1])
    self.two_two_types = np.array([1, 1, 2, 2])
    self.two_one_one_types = np.array([1, 1, 2, 3])
    self.neighall = neighbour.Neighbour()
    self.neigh11 = neighbour.Neighbour([1, 1])
    self.neigh22 = neighbour.Neighbour([2, 2])
    self.neigh33 = neighbour.Neighbour([3, 3])
    self.neigh12 = neighbour.Neighbour([1, 2])
    self.neigh13 = neighbour.Neighbour([1, 3])
    self.neigh23 = neighbour.Neighbour([2, 3])
    self.pair_all = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

  def test_type11(self):
    pair = self.neigh11.build_list(None, self.four_types)
    assert_equals(list(pair), self.pair_all)
    pair = self.neigh12.build_list(None, self.four_types)
    assert_equals(list(pair), [])

  def test_type12(self):
    pair = self.neigh11.build_list(None, self.two_two_types)
    assert_equals(list(pair), [(0, 1)])
    pair = self.neigh12.build_list(None, self.two_two_types)
    assert_equals(list(pair), [(0, 2), (0, 3), (1, 2), (1, 3)])
    pair = self.neigh22.build_list(None, self.two_two_types)
    assert_equals(list(pair), [(2, 3)])
    pair = self.neighall.build_list(None, self.two_two_types)
    assert_equals(list(pair), self.pair_all)

  def test_type123(self):
    pair = self.neigh11.build_list(None, self.two_one_one_types)
    assert_equals(list(pair), [(0, 1)])
    pair = self.neigh12.build_list(None, self.two_one_one_types)
    assert_equals(list(pair), [(0, 2), (1, 2)])
    pair = self.neigh22.build_list(None, self.two_one_one_types)
    assert_equals(list(pair), [])
    pair = self.neigh23.build_list(None, self.two_one_one_types)
    assert_equals(list(pair), [(2, 3)])
    pair = self.neighall.build_list(None, self.two_two_types)
    assert_equals(list(pair), self.pair_all)
