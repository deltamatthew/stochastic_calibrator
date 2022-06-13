#from ...const.const import MILLISECOND_MULTIPLIER
from .base import BrownianDiffusion, BrownianDiffusionWithMarketImpact
from ...core import CustomOrderBook
from ...core.data_input import DataInput
from ..base import StochasticProcessCalibrator
from ...const.const import MILLISECOND_MULTIPLIER

class BrownianDiffusionRolling(BrownianDiffusion, DataInput, StochasticProcessCalibrator):
    def __init__(self, dT:float, lookback_period: float, price_precision = 2, size_precision = 3, multiplier = 1):
        super().__init__(price_precision, size_precision, multiplier)
        self.dT = dT
        self.lookback_period_millisecond = lookback_period * MILLISECOND_MULTIPLIER

    def reset(self):
        self.clear_vwap()
        self.clear_dS()
        self.last_update_timestamp = 0
        self.update_params = self.update_params_init

    def read_orderbook(self, orderbook: CustomOrderBook, curr_timestamp: int):
        
        if (
            (curr_timestamp - (self.dT * MILLISECOND_MULTIPLIER) * (self.curr_vwap_timestamp // (self.dT * MILLISECOND_MULTIPLIER)) 
            >= (self.dT * MILLISECOND_MULTIPLIER))
        ):
            vwap = orderbook.calculate_vwap(depth = 1)
            if vwap is not None:
                self.append_dS_from_vwap(vwap, curr_timestamp)

    def update_params(self, curr_timestamp: int, *args, **kwargs):
        return self.update_params_init(curr_timestamp, *args, **kwargs)

    def update_params_init(self, curr_timestamp: int) -> float:
        if len(self.price_diffs) < 2:
            self.last_update_timestamp = curr_timestamp
            pass
        else:
            self.last_update_timestamp = curr_timestamp
            self.update_params = self.update_params_after_init # change of function pointer
            return self.update_params_after_init(curr_timestamp)

    def update_params_after_init(self, curr_timestamp):
        self.last_update_timestamp = curr_timestamp
        dS = self.get_dS_series(curr_timestamp, self.lookback_period_millisecond)
        self._sigma = self.estimate_sigma(dS, self.dT)
        return self.model_params

class BrownianDiffusionWithMarketImpactRolling(
    BrownianDiffusionWithMarketImpact, 
    BrownianDiffusionRolling
    ):
    """
        Not ready
    """
    def read_trade(self, trade, curr_timestamp):
        self.add_trade_count(trade, curr_timestamp)
