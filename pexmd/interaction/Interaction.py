"""
Main Interaction module.
"""

import numpy as np
import itertools as it

class Interaction(object):
  """
  Base Interaction class.
  """
  def __init__(self):
    pass

  def forces(self, x, v, pairs=None):
    """
    Main loop calculation.

    NOTE: This is highly experimental and slow.
    It is just meant to be a proof of concept for the main loop, but
    has to change *dramatically*, even in arity, when we want to add
    lists of neighbors, parallelization and so on.
    """
    return np.zeros_like(x), 0.0

class ShortRange(Interaction):
  """
  Base short-range class
  """
  def __init__(self, rcut, shift_style='None'):
    """
    Base short-range class

    Parameters
    ----------

    rcut : float
        The cut radius parameter

    shift_style: {'None', 'Displace', 'Splines'}
        Shift style when approaching rcut

    .. note:: 'Splines' not implemented yet
    """
    self.rcut = rcut
    self.shift_style = shift_style
    super().__init__()

  def forces(self, x, v, pairs=None):
    """
    Calculate Lennard-Jones force
    """
    energ = 0
    forces = np.zeros_like(x)
    if pairs == None:
      pairs = np.array(list(it.combinations(range(len(x)), 2)), dtype=np.int64)
    for i, j in pairs:
      f = self.pair_force(x[i], x[j])
      energ += self.pair_energ(x[i], x[j])
      forces[i] += f
      forces[j] -= f
    return forces, energ

  def pair_force(self, s1, s2):
    return np.array([0, 0, 0], dtype=np.float32)

  def pair_energ(self, s1, s2):
    return 0.0

class LennardJones(ShortRange):
  """
  Lennard-Jones potential
  """
  def __init__(self, rcut, eps, sigma, shift_style='None'):
    self.eps = eps
    self.sigma = sigma
    super().__init__(rcut, shift_style)


  def pair_force(self, s1, s2):
    d = np.linalg.norm(s1-s2)
    if d > self.rcut:
      return np.zeros_like(s1)
    ljf = 24*self.eps*(2*self.sigma**12/d**14 - self.sigma**6/d**8)*(s1-s2)
    if self.shift_style == 'None':
      return ljf
    elif self.shift_style == 'Displace':
      return ljf

  def pair_energ(self, s1, s2):
    vcut = 4*self.eps*(self.sigma**12/self.rcut**12 - self.sigma**6/self.rcut**6)
    d = np.linalg.norm(s1-s2)
    if d >= self.rcut:
      return 0
    ljf = 4*self.eps*(self.sigma**12/d**12 - self.sigma**6/d**6)
    if self.shift_style == 'None':
      return ljf
    elif self.shift_style == 'Displace':
      return ljf - vcut
