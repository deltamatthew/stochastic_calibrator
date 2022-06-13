from quant.market import Orderbook as ExchangeOrderBook
from quant.utils import logger

ABNORMAL_PCT_THRESHOLD = 0.01

class CustomOrderBook(ExchangeOrderBook):
    def __init__(self, orderbook,):
        self._orderbook = orderbook
        #super().__init__(orderbook.platform, orderbook.symbol, orderbook.asks, orderbook.bids, orderbook.timestamp)
        self.asks_depth = len(self.asks) if self.asks else 0
        self.bids_depth = len(self.bids) if self.bids else 0
        self._is_abnormal_spread = self.detect_abnormal_spread()

    def get_bid_price(self, depth_num):
        if self.bids_depth < depth_num: raise Exception(f"Bids depth is not enough to fetch {depth_num} depth.")
        return float(self.bids[depth_num][0])

    def get_ask_price(self, depth_num):
        if self.asks_depth < depth_num: raise Exception(f"Asks depth is not enough to fetch {depth_num} depth.")
        return float(self.asks[depth_num][0])

    def get_bid_size(self, depth_num):
        if self.bids_depth < depth_num: raise Exception(f"Bids depth is not enough to fetch {depth_num} depth.")
        return float(self.bids[depth_num][1])

    def get_ask_size(self, depth_num):
        if self.asks_depth < depth_num: raise Exception(f"Asks depth is not enough to fetch {depth_num} depth.")
        return float(self.asks[depth_num][1])
    
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
        if self.bids_depth < depth or self.asks_depth < depth: return None
        for i in range(depth):
            total_amt += 1 / float(self.bids[i][1]) + 1 / float(self.asks[i][1])
            dollar_value += float(self.bids[i][0]) / float(self.bids[i][1]) + float(self.asks[i][0]) / float(self.asks[i][1])
        vwap = dollar_value / total_amt
        return vwap

    def detect_abnormal_spread(self):
        if not self.best_bid or not self.best_bid:
            self._is_abnormal_spread = False
        best_bid = self.best_bid
        best_ask = self.best_ask
        if (best_ask - best_bid)/((best_ask + best_bid)/2) > ABNORMAL_PCT_THRESHOLD:
            self._is_abnormal_spread = True
        else:
            self._is_abnormal_spread = False

    @property
    def is_abnormal_spread(self):
        return self._is_abnormal_spread

    @property
    def asks(self):
        return self._orderbook.asks
    
    @property
    def bids(self):
        return self._orderbook.bids

    @property
    def platform(self):
        return self._orderbook.platform

    @property
    def symbol(self):
        return self._orderbook.symbol

    @property
    def action(self):
        return self._orderbook.action

    @property
    def price(self):
        return self._orderbook.price

    @property
    def quantity(self):
        return self._orderbook.quantity

    @property
    def timestamp(self):
        return self._orderbook.timestamp

    @property
    def best_bid(self):
        return self.get_bid_price(0)

    @property
    def best_ask(self):
        return self.get_ask_price(0)
    