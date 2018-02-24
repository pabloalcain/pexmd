"""
Main Integrator module
"""

import numpy as np

class Integrator(object):
  """
  Integrator base class
  """

  def __init__(self, dt):
    self.dt = dt

  def first_step(self, x, v, a):
    return x, v

  def last_step(self, x, v, a):
    return x, v

class NVE(Integrator):
  """
  NVE base class
  """

class VelVerlet(NVE):
  """
  Velocity Verlet Integrator
  """

  def first_step(self, x, v, a):
    x = x + v*self.dt + 0.5*a*self.dt**2
    v = v + 0.5*a*self.dt
    return x, v

  def last_step(self, x, v, a):
    v = v + 0.5*a*self.dt
    return x, v

class NVT(Integrator):
  """
  NVT base class
  """
  def __init__(self, dt, temperature):
    self.temperature = temperature
    super().__init__(dt)
