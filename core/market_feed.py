import gc
import pandas as pd

from quant.utils import tools

class MarketFeed:
    def __init__(self, platform, symbol, config):
        """
        Class to store config of a specific traded pair of asset in exchange and the data format needed for Avellaneda-Stoikov model
        Args:
            platform:   platform of the traded pairs
            symbol:     symbol of traded pairs in that platform
            config:     configuration of the market information and the specification of how long the data would be stored
                        example of config:
                            {
                                    "platform": "binance-futures",
                                    "symbol": "ETH/USDT",
                                    "leverage": 20,
                                    "waiting_dt":10,
                                    "sleep_dt": 0.5,
                                    "sigma_multiplier": 1.5, 
                                    "gamma": 0.01,
                                    "max_qty": 20,
                                    "is_mk_on": true,
                                    "liquidation_factor": 1,
                                    "size_precision": 3,
                                    "price_precision": 2,
                                    "rebate": 0.0001,
                                    "tick_size":0.01,
                                    "step":5,
                                    "lambda_sampling_freq": 1,
                                    "sigma_freq": 1,
                                    "lambda_tol": 0.5,
                                    "delta_max": 2.0,
                                    "trades_params":{
                                        "lookback_period": 21600,
                                        "update_freq": 3600
                                    },
                                    "diffusion_params": {
                                        "lookback_period": 3600
                                    }
                                }
        """
        # from config
        self._platform = platform
        self._symbol = symbol
        self.orderbook = {"asks": [], "bids": [], "timestamp": tools.get_cur_timestamp_ms()}
        self.config = config
        self._place_order_time = None

        self.curr_vwap = None
        self.curr_vwap_timestamp = None
        self.vwaps = []
        self.vwap_timestamps = [] #tools.get_cur_timestamp_ms()
        self.deltas = []
        self.delta_timestamps = []
        self.init_success = None
        self.num_buy_trades = 0
        self.num_sell_trades = 0

    @staticmethod
    def filter_data(timestamp: int, series: pd.Series) -> pd.Series:
        """
            Filter series according to its timestamp index
            # timestamp is in ms
        """ 
        series = series.loc[series.index > timestamp]
        gc.collect()
        return series

    @property
    def place_order_time(self):
        return self._place_order_time

    @place_order_time.setter
    def set_place_order_time(self, place_order_time):
        self._place_order_time = place_order_time

    def set_curr_vwap(self, vwap:float):
        self.curr_vwap = vwap

    def set_curr_vwap_timestamp(self, vwap_timestamp:int):
        self.curr_vwap_timestamp = vwap_timestamp

    def add_vwap(self, vwap:float, vwap_timestamp: int) -> None:
        """
        Function to append vwap in this class

        Args:
            vwap (float): weighted price in LOB
            vwap_timestamps (int): timestamp for the data
        """
        self.vwaps.append(vwap)
        self.vwap_timestamps.append(vwap_timestamp)


    def add_delta(self, delta: float, timestamp: int) -> None:
        """
        Function to append delta in this class

        Args:
            delta (float): distance between traded price and the latest reference price retrieved in LOB data
            timestamp (int): timestamp for the data
        """        
        self.deltas.append(delta)
        self.delta_timestamps.append(timestamp)
    
    def get_vwap_series(self, curr_timestamp: int, lookback_period: float) -> pd.Series:
        """
        Function to drop data exceed lookback period and get weighted average price as pd.Series form stored

        Args:
            timestamp (int): current timestamp
        """
        if self.vwaps:
            series = pd.Series(self.vwaps, index = self.vwap_timestamps)
            series = self.filter_data(curr_timestamp - lookback_period, series)
            self.vwaps = series.to_list()
            self.vwap_timestamps = series.index.to_list()
            gc.collect()
            return series
        return pd.Series(dtype=float)
    
    def get_delta_series(self, curr_timestamp: int, lookback_period: float) -> pd.Series:
        """
        Function to drop data exceed lookback period and get weighted average price as pd.Series form stored

        Args:
            timestamp (int): current timestamp
        """        
        if self.deltas and self.delta_timestamps:
            series = pd.Series(self.deltas, index = self.delta_timestamps)
            series = self.filter_data(curr_timestamp - lookback_period, series)
            self.deltas = series.to_list()
            self.delta_timestamps = series.index.to_list()
            series.index = pd.to_datetime(self.delta_timestamps, unit="ms")
            gc.collect()
            return series
        return pd.Series(dtype=float)

    def get_vwap_timestamp(self):
        """
        Function to get the timestamp of the latest WAP 

        Returns:
            timestamp(int): the timestamp of the latest WAP
        """
        if len(self.vwap_timestamps) > 0:
            return self.vwap_timestamps[-1]
        return 0

    def calculate_vwap(self, depth:int = 1):
        """
        Function to calculate weighted price 

        Args:
            depth (int, optional): Number of orderbook depth that used. Defaults to 5.
        Returns:
            wap: weighted average price from the first 5 layer of LOB
        """
        # consider only the first n levels
        total_amt = 0
        dollar_value = 0
        if self.orderbook["bids"] and self.orderbook["asks"]:
            for i in range(depth):
                total_amt += 1/float(self.orderbook["asks"][i][1]) + 1/float(self.orderbook["bids"][i][1]) 
                dollar_value += float(self.orderbook["bids"][i][0])*(1/float(self.orderbook["bids"][i][1])) + float(self.orderbook["asks"][i][0])*(1/float(self.orderbook["asks"][i][1]))
            vwap = dollar_value/total_amt 
            return vwap
        return None
