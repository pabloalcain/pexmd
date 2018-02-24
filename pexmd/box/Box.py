"""
Main box module
"""

import numpy as np

class Box(object):
  """
  Box class
  """
  def __init__(self, x0, xf, t='Periodic'):
    """
    Parameters
    ----------

    x0 : NumPy array
        Initial vertex of the box

    xf : NumPy array
        Final vertex of the box

    t : {'Periodic', 'Fixed'}
        Type of boundary
    """
    self.x0 = np.array(x0)
    self.xf = np.array(xf)
    self.t = t

  def wrap_boundary(self, x, v):
    """
    Apply boundary conditions

    Parameters
    ----------

    x, v : NumPy array
        Positions and velocities of the particles

    Returns
    -------

    x, v : NumPy array
        Positions and velocities updated
    """
    x_b = np.copy(x)
    v_b = np.copy(v)
    if self.t == 'Periodic':
      l = self.xf - self.x0
      for i, pos in enumerate(x):
        for j, p in enumerate(pos):
          while p > self.xf[j]:
            p -= l[j]
          while p < self.x0[j]:
            p += l[j]
          x_b[i, j] = p
    elif self.t == 'Fixed':
      l = self.xf - self.x0
      for i, pos in enumerate(x):
        for j, p in enumerate(pos):
          m = 1
          while (p > self.xf[j]) or (p < self.x0[j]):
            if p > self.xf[j]:
              m *= -1
              p = 2*self.xf[j] - p
            if p < self.x0[j]:
              p = 2*self.x0[j] - p
              m *= -1
          x_b[i, j] = p
          v_b[i, j] *= m
    return x_b, v_b
