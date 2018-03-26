"""
Main box module
"""

import numpy as np
import ctypes as ct

box = ct.CDLL('pexmd/box/box.so')
boxperiodic_c = box.periodic
boxperiodic_c.argtypes = [ct.c_voidp, ct.c_longlong, ct.c_voidp, ct.c_voidp]
boxfixed_c = box.fixed
boxfixed_c.argtypes = [ct.c_voidp, ct.c_voidp, ct.c_longlong, ct.c_voidp, ct.c_voidp]

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
    if np.isscalar(x0):
      self.x0 = np.array([x0]*3, dtype=np.float32)
    else:
      self.x0 = np.array(x0, dtype=np.float32)
    if np.isscalar(xf):
      self.xf = np.array([xf]*3, dtype=np.float32)
    else:
      self.xf = np.array(xf, dtype=np.float32)
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
    x0p = self.x0.ctypes.data_as(ct.c_voidp)
    xfp = self.xf.ctypes.data_as(ct.c_voidp)
    xp = x.ctypes.data_as(ct.c_voidp)
    npart = len(x)
    if self.t == 'Periodic':
      boxperiodic_c(xp, npart, x0p, xfp)
    elif self.t == 'Fixed':
      vp = v.ctypes.data_as(ct.c_voidp)
      boxfixed_c(xp, vp, npart, x0p, xfp)
    return x, v
