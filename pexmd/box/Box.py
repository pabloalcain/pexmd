"""
Main box module
"""

import numpy as np
import ctypes as ct
import itertools as it

box = ct.CDLL('pexmd/box/box.so')
boxperiodic_c = box.periodic
boxperiodic_c.argtypes = [ct.c_voidp, ct.c_longlong, ct.c_voidp, ct.c_voidp]
boxfixed_c = box.fixed
boxfixed_c.argtypes = [ct.c_voidp, ct.c_voidp, ct.c_longlong, ct.c_voidp, ct.c_voidp]


def powerset(images):
  """
  Helper function for creating all the possible images.
  Will be superseeded soon and all implemented in C.
  powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)

  Parameters
  ----------

  images : iterable
      All the images in each dimension

  Returns
  -------
  all_images : iterator
      All possible combination of images

  """
  images = list(images)
  return it.chain.from_iterable(it.combinations(images, n) for n in range(1, len(images)+1))


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


  def find_ghosts(self, x, rcut):
    """
    Find ghost particles (that exist to fulfil the boundary conditions)

    Warning: this only works when rcut is less than the size of the box/2

    Parameters
    ----------

    x : NumPy array
        Positions of the particles

    rcut : float
        Cutoff radius

    Returns
    -------

    ghost_index, ghost_position : NumPy array
        Indices and positions of the ghost particles
    """

    ghost_index = []
    ghost_position = []
    # There are better ways to do this, but this will be soon implemented in C
    if self.t == 'Periodic':
      delta = self.xf - self.x0
      for i, pos in enumerate(x):
        all_delta = []
        iab = pos - self.x0 < rcut
        iarr = self.xf - pos < rcut
        d = np.zeros(3, dtype=np.int32)
        for k, is_image in enumerate(iab):
          if is_image:
            t = d.copy()
            t[k] = 1
            all_delta.append(t)
        for k, is_image in enumerate(iarr):
          if is_image:
            t = d.copy()
            t[k] = -1
            all_delta.append(t)
        for images in powerset(all_delta):
          td = sum(images)
          ghost_position.append(td*delta + x[i])
          ghost_index.append(i)
    ghost_index = np.array(ghost_index, dtype=np.int64)
    ghost_position = np.array(ghost_position, dtype=np.float32)
    return ghost_index, ghost_position
