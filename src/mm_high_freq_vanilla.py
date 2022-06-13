# -*- coding:utf-8 -*-
"""
    做市商系统
    Author: Chan Cheuk Nam
    Date: 2022/05/01
"""
import asyncio
import numpy as np
import os
from copy import deepcopy
import datetime

from ..stochastic_processes.utils.kalman_filter import AdaptiveExtendedKalmanFilter
from ..utils.helpers import calculate_alpha
from ..stochastic_processes.diffusion_processes import (
    BrownianDiffusionRolling, BrownianDiffusionAdaptive, BrownianDiffusionWithMarketImpactKalmanFilter
)
from ..stochastic_processes.market_arrival_processes import PoissonMarketArrivalProcess, PoissonMarketArrivalProcessAdaptive
from ..core.order_book import CustomOrderBook

from quant import const
from quant.config import config
from quant.utils import tools
from quant.utils import logger
from quant.utils.decorator import async_method_locker
from quant.tasks import LoopRunTask

from quant.data import OrderData
from quant.asset import Asset
from quant.market import Market
from quant.trade import Trade
from quant.order import Order
from quant.market import Orderbook
from quant.market import Trade as MarketTrade

from quant.order import ORDER_ACTION_BUY, ORDER_ACTION_SELL, ORDER_TYPE_LIMIT, \
                        ORDER_STATUS_PARTIAL_FILLED, ORDER_STATUS_FILLED, ORDER_STATUS_CANCELED, \
                        ORDER_STATUS_NEW

from pathlib import Path
import atexit
import pickle
import json

from ..const.const import (
    MILLISECOND_MULTIPLIER, ABNORMAL_PCT_THRESHOLD, 
    VANILLA_ROLLING, VANILLA_ADAPTIVE, MARKET_IMPACT_ADAPTIVE
)
MIN_ROW_TO_ESTIMATE_MARKET_ARRIVAL = 100


class MMHighFrequencyStrategy:
    def __init__(self):
        # 策略参数
        self.strategy = config.strategy
        self.market_configs = config.market_config

        self.model_params_keys = {}
        self.model_params_files = {}
        self.configs = {}
        self.orderbooks = {}
        self.init_successes = {}

        self.check_market_freq = float(config.check_market_freq)
        # data storage
        self.order_data = OrderData()
        self.saved_info = []
                
        # Class for params calibration
        self.market_arrivals = {}
        self.diffusion_processes = {}
        # TODO: throw an error if quantity is less than precision

        # maker trader
        for i, market_config in enumerate(self.market_configs):
            if market_config["is_mk_on"]:
                platform = market_config["platform"] 
                symbol = market_config["symbol"]
                # initialization
                self.initialize_market_callback(platform, symbol)
                self.initialize_orderbook(platform, symbol)
                self.initialize_success(platform, symbol)
                self.initialize_market_config(platform, symbol, market_config)
                self.initialize_market_process(platform, symbol, market_config)
                self.initialize_diffusion_process(platform, symbol, market_config)
                LoopRunTask.register(self.on_timer_update_diffusion_params, int(market_config['diffusion_params']['sampling_freq']), platform, symbol)
                LoopRunTask.register(self.on_timer_update_market_arrival_params, int(market_config['market_arrival_params']['update_freq']), platform, symbol)    
                LoopRunTask.register(self.on_timer_record_params, int(market_config['record_freq']), platform, symbol)
                self.initialize_heartbeat_functions(platform, symbol)
        
        #atexit.register(self.save_position_info)

    def set_model_params_keys(self, model_params_keys, platform, symbol):
        self.model_params_keys[platform][symbol] = model_params_keys
    
    @async_method_locker('record_params', True, timeout = None)
    async def on_timer_record_params(self, platform, symbol, *args, **kwargs):

        model_params = self.diffusion_processes[platform][symbol].model_params | self.market_arrivals[platform][symbol].model_params
        print(model_params)
        try:
            file_path = os.path.join(self.configs[platform][symbol]["params_path"], f"{platform}_{''.join(symbol.split('/'))}_{self.configs[platform][symbol]['version']}.json")
            print(file_path)
            with open(file_path, 'w') as f:
                json.dump(model_params, f)
        except Exception as e:
            print(e)
        return

    async def on_timer_place_orders(self, platform, symbol, **kwargs):
        curr_timestamp = tools.get_cur_timestamp_ms()
        await self.place_orders(platform, symbol, curr_timestamp)

    async def on_timer_check_market(self, platform, symbol, **kwargs):
        """check market
        """
        config = self.configs[platform][symbol]
        log_info = f"PLATFORM: {platform} SYMBOL: {symbol}"

        if tools.get_cur_timestamp_ms() - int(self.orderbooks[platform][symbol].timestamp) > 10000:
            if config["is_mk_on"]:
                self.configs[platform, symbol]["is_mk_on"] = False
                logger.warn(log_info + " - market stopped!", caller=self)
        else:
            if not config["is_mk_on"]:
                self.configs[platform, symbol]["is_mk_on"] = True
                logger.warn(log_info + " - market restarted!", caller=self)
    
    async def on_timer_update_diffusion_params(self, platform, symbol, **kwargs):
        curr_timestamp = tools.get_cur_timestamp_ms()
        diffusion_process = self.diffusion_processes[platform][symbol]
        diffusion_process.update_params(curr_timestamp)
        print("diffusion params are updated", diffusion_process.model_params.items())
    
    async def on_timer_update_market_arrival_params(self, platform, symbol, **kwargs):
        # for backtest rewrite this function
        curr_timestamp = tools.get_cur_timestamp_ms()
        market_arrival = self.market_arrivals[platform][symbol]
        market_arrival.update_params(curr_timestamp)
        print("market params are updated", market_arrival.model_params.items())
    
    def initialize_success(self, platform, symbol):
        if platform not in self.init_successes:
            self.init_successes[platform] = {}
        self.init_successes[platform][symbol] = None
        return 

    def initialize_orderbook(self, platform, symbol):
        if platform not in self.orderbooks:
            self.orderbooks[platform] = {}
        self.orderbooks[platform][symbol] = None
        return 

    def initialize_market_config(self, platform, symbol, market_config):
        if platform not in self.configs:
            self.configs[platform] = {}
        self.configs[platform][symbol] = market_config
        return 

    def initialize_market_process(self, platform, symbol, market_config):
        if market_config['version'] == VANILLA_ROLLING:
            if platform not in self.market_arrivals:
                    self.market_arrivals[platform] = {}
            if market_config["is_pickle_file_used"]:
                pickle_filepath = market_config["pickle_market_arrival_filepath"]
                with open(pickle_filepath, 'rb') as f:
                    self.market_arrivals[platform][symbol] = pickle.load(f)
            else:
                r = float(market_config['market_arrival_params']['initial_reference_price'])
                tick_size = float(10**(-market_config['price_precision']))
                deltas = PoissonMarketArrivalProcess.generate_deltas(r, tick_size)
                self.market_arrivals[platform][symbol] = PoissonMarketArrivalProcess(deltas, float(market_config['market_arrival_params']['lookback_period']), float(market_config['market_arrival_params']['sampling_freq']))
        elif market_config['version'] in {VANILLA_ADAPTIVE, MARKET_IMPACT_ADAPTIVE}:
            if platform not in self.market_arrivals:
                    self.market_arrivals[platform] = {}
            if market_config["is_pickle_file_used"]:
                pickle_filepath = market_config["pickle_market_arrival_filepath"]
                with open(pickle_filepath, 'rb') as f:
                    self.market_arrivals[platform][symbol] = pickle.load(f)
            else:
                r = float(market_config['market_arrival_params']['initial_reference_price'])
                tick_size = float(10**(-market_config['price_precision']))
                deltas = PoissonMarketArrivalProcessAdaptive.generate_deltas(r, tick_size)
                A = float(market_config['market_arrival_params']['initial_params']["A"])
                k = float(market_config['market_arrival_params']['initial_params']["k"])
                self.market_arrivals[platform][symbol] = PoissonMarketArrivalProcessAdaptive(deltas, dT=float(market_config['market_arrival_params']['sampling_freq']), A=A, k=k)
        
    def initialize_diffusion_process(self, platform, symbol, market_config):
        if market_config['version'] == VANILLA_ROLLING:
            if platform not in self.diffusion_processes:
                    self.diffusion_processes[platform] = {}
            if market_config["is_pickle_file_used"]:
                pickle_filepath = market_config["pickle_diffusion_filepath"]
                with open(pickle_filepath, 'rb') as f:
                    self.diffusion_processes[platform][symbol] = pickle.load(f)
            else:
                self.diffusion_processes[platform][symbol] = BrownianDiffusionRolling(float(market_config['diffusion_params']['sampling_freq']), float(market_config['diffusion_params']['lookback_period']))
        elif market_config['version'] == VANILLA_ADAPTIVE:
            if platform not in self.diffusion_processes:
                    self.diffusion_processes[platform] = {}
            if market_config["is_pickle_file_used"]:
                pickle_filepath = market_config["pickle_diffusion_filepath"]
                with open(pickle_filepath, 'rb') as f:
                    self.diffusion_processes[platform][symbol] = pickle.load(f)
            else:
                diffusion_params = market_config['diffusion_params']
                diffusion_sampling_freq = diffusion_params['sampling_freq']
                diffusion_decaying_length = diffusion_params['decaying_length']
                diffusion_required_prop = diffusion_params["required_prop"]
                diffusion_decaying_length_in_bin = diffusion_decaying_length / diffusion_sampling_freq
                alpha = calculate_alpha(length = diffusion_decaying_length_in_bin, required_p = diffusion_required_prop)
                dT = float(market_config['diffusion_params']['sampling_freq'])
                sigma20 = float(market_config['diffusion_params']['initial_params']["sigma20"])
                size_precision = int(market_config["size_precision"])
                price_precision = int(market_config["price_precision"])
                self.diffusion_processes[platform][symbol] = BrownianDiffusionAdaptive(dT=dT, sigma20=sigma20, alpha=alpha, price_precision=price_precision, size_precision=size_precision, multiplier=market_config['multiplier'])

        elif market_config['version'] == MARKET_IMPACT_ADAPTIVE:
            if platform not in self.diffusion_processes:
                self.diffusion_processes[platform] = {}
            if market_config["is_pickle_file_used"]:
                pickle_filepath = market_config["pickle_diffusion_filepath"]
                with open(pickle_filepath, 'rb') as f:
                    self.diffusion_processes[platform][symbol] = pickle.load(f)

            else:
                diffusion_params = market_config['diffusion_params']
                kalman_filter_params = diffusion_params['kalman_filter_params']
                diffusion_sampling_freq = diffusion_params['sampling_freq']
                diffusion_decaying_length = diffusion_params['decaying_length']
                diffusion_required_prop = diffusion_params["required_prop"]
                diffusion_decaying_length_in_bin = diffusion_decaying_length / diffusion_sampling_freq
                alpha = calculate_alpha(length = diffusion_decaying_length_in_bin, required_p = diffusion_required_prop)
                decay_factor = 1 - alpha

                akf = AdaptiveExtendedKalmanFilter(curr_measurement_noise_cov = kalman_filter_params['curr_measurement_noise_cov'],
                                                    curr_state = kalman_filter_params['curr_state'],
                                                    curr_state_estimate_cov = kalman_filter_params['curr_state_estimate_cov'],
                                                    curr_state_noise_cov = kalman_filter_params['curr_state_noise_cov'],
                                                    measurement_decay=decay_factor, state_decay=decay_factor, 
                                                    gamma = kalman_filter_params['gamma'])
                
                dT = float(market_config['diffusion_params']['sampling_freq'])
                sigma20 = float(market_config['diffusion_params']['initial_params']["sigma20"])
                size_precision = int(market_config["size_precision"])
                price_precision = int(market_config["price_precision"])
                self.diffusion_processes[platform][symbol] = BrownianDiffusionWithMarketImpactKalmanFilter(kalman_filter=akf, dT=dT, sigma20=sigma20,alpha=alpha,
                                                                                                            price_precision=price_precision,
                                                                                                            size_precision=size_precision,
                                                                                                            multiplier=market_config['multiplier'])
        else:
            raise Exception('Version not defined')

    def initialize_market_callback(self, platform, symbol):
        Market(const.MARKET_TYPE_TRADE, platform, symbol, self.on_event_trade_callback)
        Market(const.MARKET_TYPE_ORDERBOOK, platform, symbol, self.on_event_orderbook_callback)


    def initialize_heartbeat_functions(self, platform, symbol):
        LoopRunTask.register(self.on_timer_check_market, self.check_market_freq, platform, symbol)

    def save_position_info(self):
        import pickle
        for market_config in self.market_configs:
            platform = str(market_config["platform"])
            symbol = str(market_config["symbol"])
            with open(Path(market_config["pickle_market_arrival_path"]) / 'logs' / f"{platform}_{symbol.replace('/', '')}_market_arrival.pickle", 'wb') as f: pickle.dump(self.market_arrivals[platform][symbol], f)
            with open(Path(market_config["pickle_diffusion_path"]) / 'logs' / f"{platform}_{symbol.replace('/', '')}_diffusion.pickle", 'wb') as f: pickle.dump(self.diffusion_processes[platform][symbol], f)

                    
    async def on_event_maker_init_success_callback(self, success, error, **kwargs):
        """ maker init
        """
        logger.info(kwargs, caller=self)
        logger.info("success: ", success, " error:", error, caller=self)
        platform = kwargs["platform"] 
        if platform == const.EMULATOR or platform == const.BACK_TESTER:
            for platform, symbols in self.init_successes.items():
                for symbol in symbols:
                    self.init_successes[platform][symbol] = success
        symbol = kwargs["symbol"]
        platform = platform.split('@')[0] # backtest/emulator
        self.init_successes[platform][symbol] = success


    @async_method_locker('orderbook_callback', False)
    async def on_event_orderbook_callback(self, orderbook: Orderbook):
        """orderbook callback
        """
        # record orderbook
        platform = orderbook.platform
        platform = platform.split('@')[0] # backtest/emulator
        symbol = orderbook.symbol
    
        # update deltas
        curr_timestamp = int(orderbook.timestamp)
        self.orderbooks[platform][symbol] = CustomOrderBook(orderbook)
        orderbook = self.orderbooks[platform][symbol]
        if orderbook.is_abnormal_spread:
            return 

        self.diffusion_processes[platform][symbol].read_orderbook(orderbook, curr_timestamp)
        self.market_arrivals[platform][symbol].read_orderbook(orderbook, curr_timestamp)

    @async_method_locker('trade_callback', True, timeout = None)
    async def on_event_trade_callback(self, trade: MarketTrade):
        """trade callback
        """
        # record orderbook
        platform = trade.platform
        platform = platform.split('@')[0] #backtest/emulator
        symbol = trade.symbol
        curr_timestamp = int(trade.timestamp)
        market_arrival = self.market_arrivals[platform][symbol]
        market_arrival.read_trade(trade, curr_timestamp)
        diffusion_process = self.diffusion_processes[platform][symbol]
        diffusion_process.read_trade(trade, curr_timestamp)
