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
    return np.zeros_like(x)

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
    # I have to split it to avoid double-counting. Don't want to get
    # too fancy since it will change when creating neighbor lists
    if self.types[0] == self.types[1]:
      for i, s1 in enumerate(x1):
        for j, s2 in enumerate(x2[i+1:]):
          f = self._lj(s1, s2)
          ii = i1[i]
          jj = i2[j+i+1]
          forces[ii] += f
          forces[jj] -= f
    else:
      for i, s1 in enumerate(x1):
        for j, s2 in enumerate(x2):
          f = self._lj(s1, s2)
          ii = i1[i]
          jj = i2[j]
          forces[ii] += f
          forces[jj] -= f
    return forces


  def _lj(self, s1, s2):
    d = np.linalg.norm(s1-s2)
    if d > self.rcut:
      return np.zeros_like(s1)
    ljf = 24*self.eps*(2*self.sigma**12/d**13 - self.sigma**6/d**7)*(s1-s2)
    if self.shift_style == 'None':
      return ljf
    elif self.shift_style == 'Displace':
      return ljf
