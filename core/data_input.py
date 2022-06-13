import abc
import gc
import pandas as pd
import abc
from .order_book import CustomOrderBook
from quant.market import Trade

class DataInput(abc.ABC):
    def __init__(self, price_precision = 2, size_precision = 3, multiplier = 1):
        self.last_vwap = None
        self.curr_vwap = None
        self.curr_vwap_timestamp = 0
        self.price_diffs = []
        self.price_diffs_timestamps = [] #tools.get_cur_timestamp_ms()
        self.empirical_deltas = []
        self.empirical_delta_timestamps = []
        self.num_buy_trades = 0
        self.num_sell_trades = 0
        self.trade_imbalances = []

        self.tick_size = 10**(-price_precision)
        self.min_qty = 10**(-size_precision) / multiplier

    def read_trade(self, trade: Trade, curr_timestamp):
        pass

    def read_orderbook(self, orderbook: CustomOrderBook, curr_timestamp):
        """
            the operation used must be of form func(vwap, curr_timestamp)
        """
        pass

    def add_trade_count(self, trade: Trade, curr_timestamp):
        if float(trade.quantity) > self.min_qty:
            if trade.action == "BUY":
                self.num_buy_trades += 1
            elif trade.action == "SELL":
                self.num_sell_trades += 1

    def calculate_trade_imbalance(self):
        return self.num_buy_trades - self.num_sell_trades

    def append_trade_imbalance(self, ):
        self.trade_imbalances.append(self.calculate_trade_imbalance())

    def append_empirical_delta(self, trade: Trade, timestamp: int) -> None:
        """
        Function to append empirical_delta in this class

        Args:
            empirical_delta (float): distance between traded price and the latest reference price retrieved in LOB data
            timestamp (int): timestamp for the data
        """
        empirical_delta = abs(float(trade.price) - self.curr_vwap)
        self.empirical_deltas.append(empirical_delta)
        self.empirical_delta_timestamps.append(timestamp)

    def get_empirical_delta_series(self, curr_timestamp: int, lookback_period: float) -> pd.Series:
        """
        Function to drop data exceed lookback period and get weighted average price as pd.Series form stored

        Args:
            timestamp (int): current timestamp
        """        
        if self.empirical_deltas and self.empirical_delta_timestamps:
            series = pd.Series(self.empirical_deltas, index = self.empirical_delta_timestamps)
            series = self.filter_data(curr_timestamp - lookback_period, series)
            self.empirical_deltas = series.to_list()
            self.empirical_delta_timestamps = series.index.to_list()
            series.index = pd.to_datetime(self.empirical_delta_timestamps, unit="ms")
            return series
        return pd.Series(dtype=float)
    
    def clear_trade_imbalance(self):
        self.num_buy_trades = 0
        self.num_sell_trades = 0

    def clear_empirical_deltas(self):
        self.empirical_deltas.clear()
        self.empirical_delta_timestamps.clear()

    def clear_dS(self):
        self.price_diffs.clear()
        self.price_diffs_timestamps.clear()

    def clear_vwap(self):
        self.curr_vwap = None
        self.last_vwap = None
        self.curr_vwap_timestamp = 0

    @staticmethod
    def filter_data(timestamp: int, series: pd.Series) -> pd.Series:
        """
            Filter series according to its timestamp index
            # timestamp is in ms
        """ 
        series = series.loc[series.index > timestamp]
        return series

    def set_vwap_and_timestamp(self, vwap: float, curr_timestamp: int):
        self.curr_vwap = vwap
        self.curr_vwap_timestamp = curr_timestamp

    def append_dS_from_vwap(self, vwap: float, curr_timestamp):
        if self.curr_vwap is not None:
            self._append_dS(self._calculate_dS(vwap), curr_timestamp)
        self.set_vwap_and_timestamp(vwap, curr_timestamp)

    def _calculate_dS(self, vwap: float):
        return vwap - self.curr_vwap

    def _append_dS(self, dS:float, vwap_timestamp: int) -> None:
        """
        Function to append vwap in this class

        Args:
            vwap (float): weighted price in LOB
            vwap_timestamps (int): timestamp for the data
        """
        self.price_diffs.append(dS)
        self.price_diffs_timestamps.append(vwap_timestamp)

    def get_dS_series(self, curr_timestamp: int, lookback_period: float) -> pd.Series:
        """
        Function to drop data exceed lookback period and get weighted average price as pd.Series form stored

        Args:
            timestamp (int): current timestamp
        """
        if self.price_diffs:
            series = pd.Series(self.price_diffs, index = self.price_diffs_timestamps)
            series = self.filter_data(curr_timestamp - lookback_period, series)
            self.price_diffs = series.to_list()
            self.price_diffs_timestamps = series.index.to_list()
            return series
        return pd.Series(dtype=float)

    def get_vwap_timestamp(self):
        return self.curr_vwap_timestamp

    def get_dS_timestamp(self):
        """
        Function to get the timestamp of the latest WAP 

        Returns:
            timestamp(int): the timestamp of the latest WAP
        """
        if len(self.price_diffs_timestamps) > 0:
            return self.price_diffs_timestamps[-1]
        return 0
