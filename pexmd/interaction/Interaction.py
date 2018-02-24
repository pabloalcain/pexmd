"""
Main Interaction module.
"""

import numpy as np

class Interaction(object):
  """
  Base Interaction class.
  """
  def __init__(self, types):
    self.types = types

  def forces(self, x, v, t):
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
  def __init__(self, types, rcut, shift_style='None'):
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
    super().__init__(types)


class LennardJones(ShortRange):
  """
  Lennard-Jones potential
  """
  def __init__(self, types, rcut, eps, sigma, shift_style='None'):
    self.eps = eps
    self.sigma = sigma
    super().__init__(types, rcut, shift_style)

  def forces(self, x, v, t):
    """
    Calculate Lennard-Jones force
    """
    x1 = x[t == self.types[0]]
    x2 = x[t == self.types[1]]
    i1 = np.arange(len(x))[t == self.types[0]]
    i2 = np.arange(len(x))[t == self.types[1]]
    forces = np.zeros_like(x)
    energ = 0
    # I have to split it to avoid double-counting. Don't want to get
    # too fancy since it will change when creating neighbor lists
    if self.types[0] == self.types[1]:
      for i, s1 in enumerate(x1):
        for j, s2 in enumerate(x2[i+1:]):
          f = self.pair_force(s1, s2)
          ii = i1[i]
          jj = i2[j+i+1]
          forces[ii] += f
          forces[jj] -= f
          energ += self.pair_energ(s1, s2)
    else:
      for i, s1 in enumerate(x1):
        for j, s2 in enumerate(x2):
          f = self.pair_force(s1, s2)
          ii = i1[i]
          jj = i2[j]
          forces[ii] += f
          forces[jj] -= f
          energ += self.pair_energ(s1, s2)
    return forces, energ


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
