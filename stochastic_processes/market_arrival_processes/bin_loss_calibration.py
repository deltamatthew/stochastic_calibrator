import numpy as np
from typing import List, Tuple
from scipy.optimize import basinhopping

from .utils.minimizer import BoundsInclusive, TakeStep
from .base import PoissonMarketArrivalProcess, MIN_ROW_TO_ESTIMATE_MARKET_ARRIVAL

class PoissonMarketArrivalProcessWithBinLossFunction(PoissonMarketArrivalProcess):
    BOUNDS = [(1e-10, np.inf)]
    GLOBAL_MIN_BOUND = BoundsInclusive(BOUNDS)
    PRESET_X0 = 0.5

    def update_params(self, curr_timestamp, x0: float = None):
        self.last_update_timestamp = curr_timestamp        
        empirical_deltas = self.get_empirical_delta_series(curr_timestamp, self.lookback_period * 1000)
        if (len(empirical_deltas) > MIN_ROW_TO_ESTIMATE_MARKET_ARRIVAL):
            filtered_deltas, filtered_lambda_hat = self._estimate_lambdas(
                empirical_deltas,
                self.deltas, 
                self.dT
            )
            self.A, self.k = self.estimate_params(
                filtered_deltas, 
                filtered_lambda_hat, 
                x0 if x0 is not None
                else self.k if self.k is not None
                else self.PRESET_X0)
        return self.model_params

    @classmethod
    def estimate_params(cls, deltas, lambdas_hat, x0: List[float], **kwargs):
        MIN_P = .00005
        lambdas_diff = lambdas_hat[:-1] - lambdas_hat[1:]
        deltas_bin = np.column_stack((deltas[:-1], deltas[1:]))
        filtered_index = lambdas_diff > np.sum(lambdas_diff) * MIN_P
        deltas_bin = deltas_bin[filtered_index]
        lambdas_diff = lambdas_diff[filtered_index]
        A = lambdas_hat[0]
        return cls.estimate_params_global_min(x0 = x0, r_arg=(A, deltas_bin, lambdas_diff),
                                        minimizer_params={'niter': kwargs['niter'], 'stepsize': kwargs['stepsize']})

    def estimate_params_global_min(self, x0: float, r_arg: Tuple, 
                    minimizer_params: dict = {'niter': 40, 'stepsize': 0.5}) -> Tuple[float, float]:
        """
        input initial guess, sigma and abs_delta as information and hyperparameters to find A and k that minimize loss function r

        Args:
            x0: initial guess of k
            r_args: arguments for r
            minimizer_params: dictionary of parameters in the global minimizer for loss function, Defaults to { 'niter': 40, 'stepsize': 0.5 }.

        Returns:
            k
        """
        #scaled_sigma =  sigma0*np.sqrt(dt) # sigma should be the same for all trades inbetween
        minimizer_kwargs = {"method":"L-BFGS-B", "jac": False, 'args': r_arg}
        takestep = TakeStep(minimizer_params['stepsize'])
        res = basinhopping(self.r, x0, minimizer_kwargs=minimizer_kwargs,niter=minimizer_params['niter'], 
                            take_step=takestep, accept_test=self.GLOBAL_MIN_BOUND) #global minimizer
        return res['x']

    @staticmethod
    def r(k, A, deltas_bin: np.ndarray, lambdas_diff: np.ndarray,):
        """
        calibration loss function considering bins

        Args:
            x: [A, k]
            deltas: numpy 1D array of deltas(distances from reference price)
            lambdas_hat: numpy 1D array of lambdas_hat(estimated intensity)
        Returns:
            loss: value to be minimized
        """
        return sum(
            abs(
                (A * (np.exp(-k * deltas_bin[:,0]) - np.exp(- k * deltas_bin[:,1])) - lambdas_diff)*(deltas_bin[:,1]-deltas_bin[:,0])
            )
        )