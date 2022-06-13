from .base import   BrownianDiffusion, BrownianDiffusionWithTrend, BrownianDiffusionWithMarketImpact
from .regular_calibration import BrownianDiffusionWithMarketImpactRolling
from ...core.data_input import DataInput
from ...core import CustomOrderBook
from ..base import StochasticProcessCalibrator
import numpy as np
from ...const.const import MILLISECOND_MULTIPLIER

class BrownianDiffusionAdaptive(BrownianDiffusion, DataInput, StochasticProcessCalibrator):
    def __init__(self, dT: float, sigma20: float, alpha: float, price_precision = 2, size_precision = 3, multiplier = 1):
        super().__init__(price_precision, size_precision, multiplier)
        self.dT = dT
        self.alpha = alpha
        self.sigma2 = sigma20
        self._sigma = np.sqrt(sigma20)

    def reset(self):
        """
            method to reset only the latest data of vwap, used for resetting class without modifying sigma
        """
        self.clear_vwap()
        self.last_update_timestamp = 0
        self.update_params = self.update_params_init

    def read_orderbook(self, orderbook: CustomOrderBook, curr_timestamp: int):
        if (
            (
                curr_timestamp - 
                (self.dT * MILLISECOND_MULTIPLIER) * 
                (
                    self.curr_vwap_timestamp // 
                    (self.dT * MILLISECOND_MULTIPLIER)
                )
            )
            >= (self.dT * MILLISECOND_MULTIPLIER)
        ):        
            vwap = orderbook.calculate_vwap(depth = 1)
            if vwap is not None:
                self.set_vwap_and_timestamp(vwap, curr_timestamp)

    def update_params(self, curr_timestamp, *args, **kwargs):
        return self.update_params_init(curr_timestamp, *args, **kwargs)

    def update_params_init(self, curr_timestamp, *args, **kwargs):
        if self.last_vwap is None:
            if self.curr_vwap is not None:
                self.last_vwap = self.curr_vwap
                self.last_update_timestamp = curr_timestamp
                return self.model_params
        else:
            self.update_params = self.update_params_after_init
            return self.update_params_after_init(curr_timestamp, *args, **kwargs)

    def update_params_after_init(self, curr_timestamp: int, *args, **kwargs):
        dS = self.curr_vwap - self.last_vwap
        self.update_sigma(dS)
        # mark to the last record
        self.last_update_timestamp = curr_timestamp
        self.last_vwap = self.curr_vwap
        return self.model_params

    def update_sigma(self, dS: float):
        self.sigma2 = self.alpha * dS**2 / self.dT + (1 - self.alpha) * self.sigma2
        self._sigma = np.sqrt(self.sigma2)

class BrownianDiffusionWithMarketImpactKalmanFilter(
        BrownianDiffusionWithMarketImpact,
        DataInput,
        StochasticProcessCalibrator
    ):
    def __init__(self, kalman_filter, dT: float, sigma20:float, alpha: float, price_precision = 2, size_precision = 3, multiplier = 1):
        super().__init__(price_precision, size_precision, multiplier)
        self.dT = dT
        self.alpha = alpha
        self.sigma2 = sigma20
        self.kalman_filter = kalman_filter

    def reset(self):
        """
            method to reset only the latest data of vwap, used for resetting class without modifying sigma
        """
        self.clear_vwap()
        self.clear_trade_imbalance()
        self.last_update_timestamp = 0
        self.update_params = self.update_params_init

    def read_orderbook(self, orderbook: CustomOrderBook, curr_timestamp: int):
        vwap = orderbook.calculate_vwap(depth = 1)
        if vwap is not None:
            self.set_vwap_and_timestamp(vwap, curr_timestamp)

    def read_trade(self, trade, curr_timestamp):
        self.add_trade_count(trade, curr_timestamp)

    def update_params(self, curr_timestamp, *args, **kwargs):
        return self.update_params_init(curr_timestamp, *args, **kwargs)
        
    def update_params_init(self, curr_timestamp, *args, **kwargs):
        if self.last_vwap is None:
            if self.curr_vwap is not None:
                self.last_vwap = self.curr_vwap
                self.last_update_timestamp = curr_timestamp
                self.clear_trade_imbalance()
                return self.model_params
        else:
            self.update_params = self.update_params_after_init
            return self.update_params_after_init(curr_timestamp, *args, **kwargs)

    def update_params_after_init(self, curr_timestamp, *args, **kwargs):
        dS = self.curr_vwap - self.last_vwap
        dN = self.num_buy_trades - self.num_sell_trades
        self.update_sigma(dS - self.kalman_filter.predict(dN)[0][0])
        self.update_xi(dS, dN)
        self.last_update_timestamp = curr_timestamp
        self.last_vwap = self.curr_vwap
        self.clear_trade_imbalance()        
        return self.model_params

    def update_xi(self, dS: float, dN: float):
        self.kalman_filter.fit_once(X = dN, y = dS)
        self._xi = self.kalman_filter.curr_state[0][0]

    def update_sigma(self, dS: float):
        self.sigma2 = self.alpha * dS**2 / self.dT + (1 - self.alpha) * self.sigma2
        self._sigma = np.sqrt(self.sigma2)