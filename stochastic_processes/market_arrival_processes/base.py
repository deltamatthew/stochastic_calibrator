import pandas as pd
import numpy as np
from scipy.optimize import basinhopping
from sklearn.linear_model import LinearRegression
from typing import Tuple, List, Callable
from itertools import repeat

from ...core.data_input import DataInput
from ..base import StochasticProcessCalibrator

from .utils import _generate_deltas, estimate_lambda_series

MIN_ROW_TO_ESTIMATE_MARKET_ARRIVAL = 100

class PoissonMarketArrivalProcess(DataInput, StochasticProcessCalibrator):
    last_update_timestamp = 0
    model_params = ['A', 'k']
    def __init__(self, deltas, lookback_period, dT = 1, A = None, k = None):
        super().__init__()        
        self.deltas = deltas
        self.dT = dT
        self.lookback_period = lookback_period
        self.A = A
        self.k = k

    @classmethod
    def from_asset_info(cls, ref_price, tick_size, lookback_period, dT = 1, A = None, k = None):
        return cls(cls.generate_deltas(ref_price, tick_size), lookback_period, dT, A, k)

    def reset(self):
        self.clear_vwap()
        self.clear_empirical_deltas()
        self.last_update_timestamp = 0

    @staticmethod
    def generate_deltas(ref_price: float, tick_size: float)->np.ndarray:
        """
            genearte deltas sereis from price data
            :params r: reference price
            :params tick_size: tick size of the traded asset
        """
        return _generate_deltas(ref_price, tick_size)

    def read_orderbook(self, orderbook, curr_timestamp: int):
        vwap = orderbook.calculate_vwap(depth = 1)
        if vwap is not None:
            self.set_vwap_and_timestamp(vwap, curr_timestamp)

    def read_trade(self, trade, curr_timestamp):
        self.append_empirical_delta(trade, curr_timestamp)

    def update_params(self, curr_timestamp):
        self.last_update_timestamp = curr_timestamp        
        empirical_deltas = self.get_empirical_delta_series(curr_timestamp, self.lookback_period * 1000)
        if (len(empirical_deltas) > MIN_ROW_TO_ESTIMATE_MARKET_ARRIVAL):
            filtered_deltas, filtered_lambda_hat = self._estimate_lambdas(
                empirical_deltas,
                self.deltas, 
                self.dT
            )
            self.A, self.k = self.estimate_params(filtered_deltas, filtered_lambda_hat)
        return self.model_params

    def estimate_lambdas(self,)->Tuple[np.ndarray, np.ndarray]:
        """
            estimate_lambdas by the information stored in this class
        """
        return self._estimate_lambdas(pd.Series(self.empirical_deltas), self.deltas, self.dT)

    @staticmethod
    def _estimate_lambdas(empirical_deltas: pd.Series, deltas: np.ndarray, dT: float,)->Tuple[np.ndarray, np.ndarray]:
        """
        Static Function to estimate lambdas and removing insignificant lambdas

        Args:
            empirical_deltas: empirical distances away from reference prices (all deltas should be > 0)
            deltas: numpy 1D array of deltas
            dT: trade count sampling frequency (measured in s)

        Returns:
            filtered_deltas, filtered_lambda_hat
        """
        LAMBDA_TOL = 1e-10 # to ensure lambda is not 0
        lambdas_hat = np.array(list(map(estimate_lambda_series, repeat(empirical_deltas), deltas, repeat(dT))))

        filtered_deltas = deltas[lambdas_hat > LAMBDA_TOL]
        filtered_lambda_hat = lambdas_hat[lambdas_hat > LAMBDA_TOL]

        return filtered_deltas, filtered_lambda_hat

    @staticmethod
    def estimate_params(deltas: np.ndarray, lambdas_hat: np.ndarray, ) -> Tuple[float, float]:
        """
        Function to estimate A and k using linear regreesion
        Model is 
            Lambda(delta) = A * np.exp(-k * delta)
        which yields
            np.log(Lambda(delta)) = np.log(A*dT) - k * delta
        hence 
            y = np.log(Lambda(delta))
            beta = (np.log(A*dT), k).T
            X = (np.ones(len(delta_arr)), delta_arr)

        Args:
            deltas: numpy 1D array of deltas(distances from reference price)
            lambdas_hat: numpy 1D array of lambdas_hat(estimated intensity)
            dT: trade count sampling frequency (measured in s)

        Returns:
            A, k: both params carry the unit of trade count sampling frequency of estimated lambdas_hat
        """
        MIN_P = .00005 # to remove very rare data
        filtered_index = (lambdas_hat > lambdas_hat.sum() * MIN_P)

        model = LinearRegression()
        model.fit(X = deltas[filtered_index].reshape(-1, 1), y = np.log(lambdas_hat[filtered_index]))
        A = np.exp(model.intercept_)
        k = -model.coef_[0]
        return A, k

    @property
    def model_params(self):
        return {'A': self.A, 'k': self.k}