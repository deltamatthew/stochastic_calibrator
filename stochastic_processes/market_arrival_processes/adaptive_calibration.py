import pandas as pd
import numpy as np
from scipy.optimize import basinhopping
from typing import Tuple, List, Callable
from itertools import repeat

from .base import PoissonMarketArrivalProcess
from quant.market import Trade
MIN_LAMBDAS_SUM = 1e-3
class PoissonMarketArrivalProcessAdaptive(PoissonMarketArrivalProcess):
    """
        poisson market arrival process arrival rate using exponential weighted moving average
    """
    def __init__(self, deltas, alpha = 0.01, lambdas = None, dT = 1, A = None, k = None):
        """                        
        >>> deltas = _generate_deltas(tick_size=tick_size, r=snapshot_df['r'].iloc[0],)

        >>> # required info
        >>> REQUIRED_INFO_PROP = 0.9
        >>> trades_decaying_length_in_bin = trades_decaying_length / trades_sampling_freq
        >>> alpha = 1 - (1 - REQUIRED_INFO_PROP)**(1 / trades_decaying_length_in_bin)
        >>> ewpap = EWMPoissonMarketArrivalProcess(deltas = deltas, alpha = alpha, dT = trades_sampling_freq)
        """
        super().__init__(deltas, lookback_period=None, dT = dT, A = A, k = k)
        self.lambdas = np.zeros_like(self.deltas) if lambdas is None else lambdas
        self.alpha = alpha

    @classmethod
    def from_asset_info(cls, ref_price, tick_size, alpha = 0.01, lambdas = None, dT = 1, A = None, k = None):
        return cls(cls.generate_deltas(ref_price, tick_size), alpha, lambdas, dT, A, k)
    
    def read_trade(self, trade, curr_timestamp):
        super().read_trade(trade, curr_timestamp)
        if curr_timestamp - (1000 * self.dT) * self.last_update_timestamp // (1000 * self.dT) >= 1:
            self.update_lambdas(self.sample_lambdas(np.array(self.empirical_deltas)))
            self.clear_empirical_deltas()
    
    def update_params(self, curr_timestamp):
        if np.sum(self.lambdas) > MIN_LAMBDAS_SUM:
            self.A, self.k = self.estimate_params(self.deltas, self.lambdas)
        self.last_update_timestamp = curr_timestamp
        return self.model_params

    def set_lambdas(self, lambdas, deltas = None):
        """
        Function used to overwrite the trades count data (lambda) stored in this class
        """
        # TODO: assert len of delta == len of lambdas
        if deltas is not None:
            assert len(lambdas) == len(deltas)
            self.deltas = deltas
        else:
            assert len(lambdas) == len(self.deltas)
        self.lambdas = lambdas

    @staticmethod
    def calculate_lambda(empirical_deltas: pd.Series, delta: float):
        return (empirical_deltas > delta).sum()

    def sample_lambdas(self, sample_empirical_deltas: pd.Series,)->float:
        """
        Function to 1-step estimate lambda using empirical deltas

        Args:
            sample_empirical_deltas: empirical distances away from reference prices (all deltas should be > 0) with time length (dT)
            delta: distance away from the reference price

        Returns:
            sum_lambdas_hat: new trades count
        """
        sample_lambdas = np.array(list(map(self.calculate_lambda, repeat(sample_empirical_deltas), self.deltas)))
        return sample_lambdas/self.dT

    def update_lambdas(self, sample_lambdas):
        self.lambdas = self.alpha * sample_lambdas + (1 - self.alpha) * self.lambdas
        return self.lambdas

    def insert_zero_lambdas(self, n_empty):
        self.lambdas = (1 - self.alpha)**n_empty * self.lambdas


class PredictivePoissonMarketArrivalProcess(PoissonMarketArrivalProcess):
    """"
        Uses multiple (2-3) EWM A and k estimation to predict A and k calibrated in a short interval of time.
        Uses Kalman Filter as the base.
    """
    def __init__(self, deltas: np.ndarray, lambdas: np.ndarray = None, 
                empirical_deltas = [], alpha1: float = 0.8, alpha2: float = .9, dT: float = 1.):
        super().__init__(deltas, dT)
        self.ewpap1 = PoissonMarketArrivalProcessAdaptive(deltas, lambdas, alpha1, dT) # A1, k1
        self.ewpap2 = PoissonMarketArrivalProcessAdaptive(deltas, lambdas, alpha2, dT) # A2, k2
        self.pap = PoissonMarketArrivalProcess(deltas, dT) # for getting true 
        self.empirical_deltas = empirical_deltas

    def estimate_params(self):
        """
            Actually it is predicting params
            Example:
            >>> A, k = ppap.estimate_params()
            >>> 
        """
        pass
    

# different loss functions
# extended A, k
def r(x: List[float], deltas: np.ndarray, lambdas_hat: np.ndarray, integrate_phi:Callable[...,float], *phi_args) -> float:
    """
    new calibration loss function

    Args:
        x: [A, k]
        deltas: numpy 1D array of deltas(distances from reference price)
        lambdas_hat: numpy 1D array of lambdas_hat(estimated intensity)
        integrate_phi: function to compute diffusion process integral
        phi_args: arguments for phi integration function

    Returns:
        loss: value to be minimized
    """
    A, k = x
    phi_args = (k, *phi_args)
    return sum((np.log(lambdas_hat) + k*deltas - np.log(A) - np.log(integrate_phi(*phi_args)))**2)

# instantaneous A_k
def r2(x, deltas: np.ndarray, lambdas_hat: np.ndarray, dT: float):
    """
    original calibration loss function

    Args:
        x: [A, k]
        deltas: numpy 1D array of deltas(distances from reference price)
        lambdas_hat: numpy 1D array of lambdas_hat(estimated intensity)
        dT: sampling frequency of trade count (measured in s)
    Returns:
        loss: value to be minimized
    """
    A, k = x
    return sum((np.log(lambdas_hat) + k*deltas - np.log(A) - np.log(dT))**2)
