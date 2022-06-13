"""
    This .py is only for the minimizer function for calibration process stated in the thesis.
"""
import pandas as pd
import numpy as np

A_BOUND = (1e-15, np.inf)
K_BOUND = (1e-15, np.inf)

# helper classes for global minimizer
class TakeStep:
    def __init__(self, stepsize):
        self.stepsize = stepsize
        self.rng = np.random.default_rng()
    def __call__(self, x):
        s = self.stepsize
        x += self.rng.uniform(-s, s,)
        return x

class BoundsInclusive:
    def __init__(self, bounds):
        self.xmin = [bound[0] for bound in bounds]
        self.xmax = [bound[1] for bound in bounds]
    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        tmin = bool(np.all(x >= self.xmin))
        tmax = bool(np.all(x <= self.xmax))
        return tmin and tmax

class BoundsExclusive:
    def __init__(self, bounds):
        self.xmin = [bound[0] for bound in bounds]
        self.xmax = [bound[1] for bound in bounds]
    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        tmin = bool(np.all(x >= self.xmin))
        tmax = bool(np.all(x <= self.xmax))
        return tmin and tmax