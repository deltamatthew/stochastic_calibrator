"""
class for minimizing function in the thesis
"""
from .base import PoissonMarketArrivalProcess
from .utils.minimizer import K_BOUND, TakeStep, BoundsInclusive

import numpy as np
from scipy.optimize import basinhopping
from typing import Callable, List, Tuple

A_BOUND = (0.0001, np.inf)
K_BOUND = (0.0001, np.inf)
class GeneralPoissonMarketArrivalProcess(PoissonMarketArrivalProcess):
    def __init__(self, r: Callable[..., float]):
        self.r = r

    def estimate_params(self, x0: List[float], r_arg: tuple, A_bound: tuple = A_BOUND, k_bound: tuple = K_BOUND, minimizer_params: dict = {'niter': 20, 'stepsize': 0.5}) -> Tuple[float, float]:
        """
        input initial guess, sigma and abs_delta as information and hyperparameters to find A and k that minimize loss function r

        Args:
            x0: initial guess of [A, k]
            r_args: arguments for r
            A_bound: minimum and maximum value for A
            k_bound: minimum and maximum value for A
            minimizer_params: dictionary of parameters in the global minimizer for loss function, Defaults to { 'niter': 20, 'stepsize': 0.5 }.

        Returns:
            A, k: both params carry the unit of trade count sampling frequency of estimated lambdas_hat
        """
        #scaled_sigma =  sigma0*np.sqrt(dt) # sigma should be the same for all trades inbetween
        minimizer_kwargs = {"method":"L-BFGS-B", "jac": False, 'args': r_arg}
        takestep = TakeStep(minimizer_params['stepsize'])
        bounds = BoundsInclusive(bounds = (A_bound, k_bound))
        res = basinhopping(self.r, x0, minimizer_kwargs=minimizer_kwargs,niter=minimizer_params['niter'], 
                            take_step=takestep, accept_test=bounds) #global minimizer
        return res['x'][0], res['x'][1] # A, k
