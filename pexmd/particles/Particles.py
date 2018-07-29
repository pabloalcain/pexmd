"""
Main Particles module.
"""

import numpy as np

class Base(object):
  """
  Base Particles class. It is abstract and we should specify which
  type of particle we actually want in order to fill it
  """
  def __init__(self, n):
    self.n = n
    self._x = np.zeros((n, 3), dtype=np.float32)
    self._v = np.zeros((n, 3), dtype=np.float32)
    self._f = np.zeros((n, 3), dtype=np.float32)
    self._t = np.zeros(n, dtype=np.int32)
    self._mass = np.zeros(n, dtype=np.float32)
    self.idx = np.arange(n)

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
    value = np.array(value, dtype=np.float32)
    number = np.shape(value)[0]
    if self.n == number:
      self._x = value
    else:
      msg = "Trying to set {0} positions for a system with {1} particles"
      raise ValueError(msg.format(number, self.n))

  @property
  def v(self):
    return self._v

  @v.setter
  def v(self, value):
    """
    Set velocities of particles.

    Parameters
    ----------

    value : 2D NumPy array
        The new velocities of particles in an Nx3 array
    """
    value = np.array(value, dtype=np.float32)
    number = np.shape(value)[0]
    if self.n == number:
      self._v = value
    else:
      msg = "Trying to set {0} velocities for a system with {1} particles"
      raise ValueError(msg.format(number, self.n))

  @property
  def f(self):
    return self._f

  @f.setter
  def f(self, value):
    """
    Set forces of particles.

    Parameters
    ----------

    value : 2D NumPy array
        The new forces of particles in an Nx3 array
    """
    value = np.array(value, dtype=np.float32)
    number = np.shape(value)[0]
    if self.n == number:
      self._f = value
    else:
      msg = "Trying to set {0} forces for a system with {1} particles"
      raise ValueError(msg.format(number, self.n))

  @property
  def a(self):
    return self._f/self._mass[:, np.newaxis]

  @property
  def t(self):
    return self._t

  @t.setter
  def t(self, value):
    """
    Set types of particles.

    Parameters
    ----------

    value : 1D NumPy array or integer
        The new types of particles in an Nx3 array
    """
    if np.isscalar(value):
      self._t = np.array([value]*self.n, dtype=np.int32)
    else:
      value = np.array(value)
      number = np.shape(value)[0]
      if self.n == number:
        self._t = value
      else:
        msg = "Trying to set {0} types for a system with {1} particles"
        raise ValueError(msg.format(number, self.n))

  @property
  def mass(self):
    return self._mass

  @mass.setter
  def mass(self, value):
    """
    Set types of particles.

    Parameters
    ----------

    value : 1D NumPy array or integer
        The new types of particles in an Nx3 array
    """
    if np.isscalar(value):
      self._mass = np.array([value]*self.n, dtype=np.float32)
    else:
      value = np.array(value)
      number = np.shape(value)[0]
      if self.n == number:
        self._mass = value
      else:
        msg = "Trying to set {0} masses for a system with {1} particles"
        raise ValueError(msg.format(number, self.n))

  def create_ghosts(self, indices, positions):
    """
    Create ghosts from indices and positions
    """
    n = len(indices)
    ghosts = Base(n)
    ghosts.idx = indices
    ghosts.x = positions
    for i, idx in enumerate(indices):
      ghosts.v[i] = self.v[idx]
      ghosts.t[i] = self.t[idx]
      ghosts.mass[i] = self.mass[idx]
    return ghosts

class PointParticles(Base):
  """
  PointParticles class.
  Typical point particles used, for example, in LJ potential.
  """
