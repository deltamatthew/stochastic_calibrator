import abc
class StochasticProcessCalibrator(abc.ABC):
    last_update_timestamp = 0
    model_params_keys = []

    @property
    @abc.abstractmethod
    def model_params(self):
        return {}

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def read_orderbook(self, orderbook, curr_timestamp):
        pass

    @abc.abstractmethod
    def read_trade(self, trade, curr_timestamp):
        pass

    @abc.abstractmethod
    def update_params(self, *args, **kwargs):
        """
            this function will be initially update_params_init, then change to update_params_after_init 
        """
        pass