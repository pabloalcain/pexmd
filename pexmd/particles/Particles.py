"""
Main Particles module
"""

import numpy as np

class Base(object):
  """
  Base Particles class. It is abstract and we should specify which
  type of particle we actually want in order to fill it
  """
  def __init__(self):
    self.n = 0
    self._x = None
    self._v = None
    self._t = None
    self._f = None
    self._mass = None

  @property
  def x(self):
    return self._x

  @x.setter
  def x(self, value):
    """
    Set positions of particles.

    Parameters
    ----------

    value : 2D NumPy array
        The new positons of particles in an Nx3 array
    """

    #Create particles if none is present
    number = np.shape(value)[0]
    if self.n == 0 or self.n == number:
      self._x = value
      self.n = number
    else:
      raise ValueError("Upa")


class PointParticles(Base):
  """
  PointParticles class.
  Typical point particles used, for example, in LJ potential.
  """
  def __init__(self):
    pass
