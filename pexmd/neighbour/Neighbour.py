#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main Interaction module.
"""

import numpy as np
import itertools as it

class Neighbour(object):
  """
  Base Neighbour class.
  """
  def __init__(self, types=None):
    self.types = types


  def build_list(self, x, t):
    """
    Build list of neighbours.
    By default, all particles of chosen types interact with each other.
    """
    if self.types == None:
      i = range(len(t))
      return np.array(list(it.combinations(i, 2)), dtype=np.int64)
    t1, t2 = self.types
    i1 = np.arange(len(t))[t == t1]
    if t1 == t2:
      return np.array(list(it.combinations(i1, 2)), dtype=np.int64)
    else:
      i2 = np.arange(len(t))[t == t2]
      return np.array(list(it.product(i1, i2)), dtype=np.int64)
