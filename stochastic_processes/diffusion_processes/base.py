import numpy as np
import pandas as pd
import abc
import re
from sklearn.linear_model import LinearRegression
import bottleneck as bn

class DiffusionProcess(abc.ABC):
    last_update_timestamp = 0

    @staticmethod
    def estimate_sigma(dS: pd.Series, dT: float) -> float:
        return dS.std() / np.sqrt(dT)

    @classmethod
    def estimate_sigmas(cls, dS: pd.Series, dT:float , lookback_window: str = '15min', is_rolling: bool = False) -> pd.Series:
        """
        Function to estimate sigma over a lookback window.

        Args:
            dS: change in prices
            dT: the sampling freq = dS
            lookback_window: the duration for estimation
            is_rolling: how the moving window moves

        Returns:
            sigmas: sigmas with dS index.
        """
        if is_rolling == True:
            sigmas = dS.rolling(lookback_window).apply(lambda x: cls.estimate_sigma(x, dT = dT))
        else:
            sigmas = dS.resample(lookback_window, label = 'right').apply(lambda x: cls.estimate_sigma(x, dT = dT,))
        sigmas = sigmas.asof(dS.index)
        return sigmas
    
    @staticmethod
    def estimate_mu(dS: pd.Series) -> float:
        return dS.mean()
    
    @staticmethod
    def estimate_xi(dS: pd.Series, dN: pd.Series) -> float:
        """
        Args:
            dS: change in prices
            dN: number of buy orders - number of sell orders
        Returns:
            xi: trade impact on reference price
        """
        model = LinearRegression(fit_intercept=False)
        model.fit(X = dN.values.reshape(-1, 1), y = dS.values.reshape(-1, 1))
        xi = model.coef_[0][0]
        return xi

    @classmethod
    def estimate_xis(cls, dS: pd.Series, dN: pd.Series, lookback_window: str = '15min', is_rolling: bool = False) -> pd.Series:
        """
        Function to estimate xis over a lookback window.

        Args:
            dS: change in prices
            lookback_window: the duration for estimation
            is_rolling: how the moving window moves

        Returns:
            xis: xis with dS index.
        """
        df_SN = pd.concat([dN.rename('dN'), dS.rename('dS')], axis = 1)
        if is_rolling == True:
            # TODO: add fast rolling regression
            def rolling_reg_without_intercept(X, y, window):
                if (X.shape == y.shape) & (len(X.shape) == 1):
                    xy = X * y
                    xx = X * X
                    beta = pd.Series(bn.move_sum(xy, window) / bn.move_sum(xx, window), index = X.index)
                    return beta
                return 
            lookback_period_seconds = int(re.search(r'\d+', lookback_window).group())
            xis = rolling_reg_without_intercept(X=dS, y=dN, window=lookback_period_seconds)
        else:
            xis = df_SN.resample(lookback_window, label = 'right').apply(lambda x: cls.estimate_xi(x["dS"], x["dN"]))
        xis = xis.asof(dS.index)
        return xis

    @abc.abstractmethod
    def integrate_phi(self, k: float, dT: float):
        """Function to be called in A_k estimation
        k: scale component for intensity
        dT: period
        
        Implementation for various processes to follow: https://www.theses.fr/2015PA066354.pdf , Chapter 4
        """
        return

    @abc.abstractmethod
    def compute_ask_delta(self):
        """Function to compute optimal ask delta from reference price
        
        Implementation for various processes to follow: https://arxiv.org/pdf/1105.3115.pdf , Chapter 5
        """
        return

    @abc.abstractmethod
    def compute_bid_delta(self):
        """Function to compute optimal bid delta from reference price
        
        Implementation for various processes to follow: https://arxiv.org/pdf/1105.3115.pdf , Chapter 5
        """
        return

    @abc.abstractmethod
    def compute_spread(self):
        """Function to compute spread which is equivate to ask_delta + bid_delta
        
        Implementation for various processes to follow: https://arxiv.org/pdf/1105.3115.pdf , Chapter 5
        """
        return 

class BrownianDiffusion(DiffusionProcess):
    model_params_keys = ['sigma', ]
    _sigma = None

    @property
    def model_params(self, ):
        return {
            'sigma': self.sigma
            }

    @property
    def sigma(self,):
        return self._sigma

    def update_params(self, *args, **kwargs):
        """
            this function will be initially update_params_init, then change to update_params_after_init 
        """
        self.update_params_init(*args, **kwargs)

    def update_params_init(self, *args, **kwargs):
        """
            to be implemented in child classes
        """
        raise NotImplementedError

    @staticmethod
    def integrate_phi(k: float, dT: float, sigma: float) -> float:
        """
        Args:
            k: scale component for intensity
            dT: scaling factor for sigma
            sigma: volatility of dS
            For eg, trade_count_freq = 10s, sigma_freq = 1s, dT = 10s
        Returns:
            phi_integral
        """
        x = (k**2)*(sigma**2)
        return (2/x)*(np.exp((x*dT)/2)-1)
    
    @staticmethod
    def compute_bid_delta(gamma:float, A:float, k:float, sigma:float, q:float, base_qty:float) -> float:
        """
        Args:
            gamma: risk adversion 
            A: shape component for intensity 
            k: scale component for intensity 
            sigma: volatility of dS
            q: running quantity
            base_qty: base quantity traded
            Note that all parameters A, k, sigma must be in the same units.
        Returns:
            bid_delta: distance away from reference for quoting bid (ref_price - bid_delta)
        """
        kp = k / base_qty
        qp = q / base_qty
        return 1 / base_qty *  1 / gamma * np.log(1 + gamma / kp) \
                +((2 * qp + 1) /2) * np.sqrt(sigma**2 * gamma / (2 * kp * A) * (1 + gamma / kp)**(1 + kp / gamma))
        
    @staticmethod
    def compute_ask_delta(gamma:float, A:float, k:float, sigma:float, q:float, base_qty: float) -> float:

        """
        Args:
            gamma: risk adversion 
            A: shape component for intensity 
            k: scale component for intensity 
            sigma: volatility of dS
            q: running quantity
            base_qty: base quantity traded
            Note that all parameters A, k, sigma must be in the same units.
        Returns:
            bid_delta: distance away from reference for quoting bid (ref_price - bid_delta)
        """
        kp = k / base_qty
        qp = q / base_qty
        return 1 / base_qty *  1 / gamma * np.log(1 + gamma / kp) \
                -((2 * qp - 1) /2) * np.sqrt(sigma**2 * gamma / (2*kp*A) * (1 + gamma / kp)**(1 + kp / gamma))

    @staticmethod
    def compute_spread(gamma:float, A:float, k:float, sigma:float, base_qty: float) -> float:
        """
        Args:
            gamma: risk adversion 
            A: shape component for intensity 
            k: scale component for intensity 
            sigma: volatility of dS
            Note that all parameters A, k, sigma must be in the same units.
        Returns:
            spread: distance between optimal bid and ask quotes (bid_delta + ask_delta)
        """
        kp = k / base_qty
        return 2 / gamma * np.log(1 + gamma / kp) + np.sqrt(sigma**2 * gamma / (2*kp*A) * (1 + gamma / kp)**(1 + kp / gamma))

class BrownianDiffusionWithTrend(BrownianDiffusion):
    model_params_keys = ['sigma', 'mu']
    _sigma = None
    _mu = None

    @property
    def model_params(self):
        return {
            'mu': self.mu,
            'sigma': self.sigma
        }
    
    @property
    def mu(self):
        return self._mu

    @staticmethod
    def integrate_phi(k: float, dT: float, mu: float, sigma: float) -> float:
        """
        Args:
            k: scale component for intensity
            dT: scaling factor for sigma
            sigma: volatility of dS
            mu: mean of dS
            For eg, trade_count_freq = 10s, sigma_freq = 1s, dT = 10s
        Returns:
            phi_integral
        """
        x = k*mu + 1/2*k**2*sigma**2
        if x == 0:
            return dT
        else:
            return 1/x*(np.exp(x*dT)-1)

    @staticmethod
    def compute_ask_delta(gamma: float, A: float, k: float,sigma: float, mu: float, q: float) -> float:
        """
        Args:
            gamma: risk adversion 
            A: shape component for intensity 
            k: scale component for intensity 
            sigma: volatility of dS
            mu: mean of dS
            q: running quantity
            Note that all parameters A, k, sigma must be in the same units.
        Returns:
            bid_delta: distance away from reference for quoting bid (ref_price - bid_delta)
        """
        return 1/gamma*np.log(1+gamma/k)-(mu/(gamma*sigma**2)-(2*q-1)/2)*np.sqrt(sigma**2*gamma/2/k/A*(1+gamma/k)**(1+k/gamma))

    @staticmethod
    def compute_bid_delta(gamma: float, A: float, k: float,sigma: float, mu: float, q: float) -> float:
        """
        Args:
            gamma: risk adversion 
            A: shape component for intensity 
            k: scale component for intensity 
            sigma: volatility of dS
            mu: mean of dS
            q: running quantity
            Note that all parameters A, k, sigma must be in the same units.
        Returns:
            bid_delta: distance away from reference for quoting bid (ref_price - bid_delta)
        """
        return 1/gamma*np.log(1+gamma/k)+(-mu/(gamma*sigma**2)+(2*q+1)/2)*np.sqrt(sigma**2*gamma/2/k/A*(1+gamma/k)**(1+k/gamma))

    @staticmethod
    def compute_spread(gamma: float, A: float, k: float,sigma: float) -> float:
        """
        Args:
            gamma: risk adversion 
            A: shape component for intensity 
            k: scale component for intensity 
            sigma: volatility of dS
            Note that all parameters A, k, sigma must be in the same units.
        Returns:
            spread: distance between optimal bid and ask quotes (bid_delta + ask_delta)
        """
        return 2/gamma*np.log(1+gamma/k)+np.sqrt(sigma**2*gamma/2/k/A*(1+gamma/k)**(1+k/gamma))

class BrownianDiffusionWithMarketImpact(DiffusionProcess):
    model_params_keys = ['sigma', 'xi']
    _sigma = None
    _xi = None

    @property
    def model_params(self):
        return {
            'xi': self._xi,
            'sigma': self._sigma
        }
    
    @property
    def xi(self):
        return self._xi

    def integrate_phi(self,  k: float, dT: float, xi: float, sigma: float) -> float:
        "Not in used"
        pass

    @staticmethod
    def compute_ask_delta(gamma: float, A: float, k: float,sigma: float, xi: float, q: float, base_qty: float) -> float:
        """
        Args:
            gamma: risk adversion 
            A: shape component for intensity 
            k: scale component for intensity 
            sigma: volatility of dS
            xi: 
            q: running quantity
            base_qty: base quantity
            Note that all parameters A, k, sigma must be in the same units.
        Returns:
            bid_delta: distance away from reference for quoting bid (ref_price - bid_delta)
        """
        xi_p = xi * base_qty
        k_p = k / base_qty
        q_p = q / base_qty
        return 1 / base_qty / gamma * np.log(1 + gamma / k_p) \
                + xi_p / (2 * base_qty) \
                -   (2 * q_p - 1) / 2 \
                    * np.exp(k_p * xi_p / 4) \
                    * np.sqrt( \
                        sigma**2 * gamma / (2 * k_p * A) \
                            * (1 + gamma / k_p)**(1 + k_p / gamma)
                        )
    
    @staticmethod
    def compute_bid_delta(gamma: float, A: float, k: float,sigma: float, xi: float, q: float, base_qty: float) -> float:
        """
        Args:
            gamma: risk adversion 
            A: shape component for intensity 
            k: scale component for intensity 
            sigma: volatility of dS
            xi: 
            q: running quantity
            base_qty: base quantity
            Note that all parameters A, k, sigma must be in the same units.
        Returns:
            bid_delta: distance away from reference for quoting bid (ref_price - bid_delta)
        """
        xi_p = xi * base_qty
        k_p = k / base_qty
        q_p = q / base_qty
        return 1 / base_qty / gamma * np.log(1 + gamma / k_p) \
                + xi_p / (2 * base_qty) \
                +   (2 * q_p + 1) / 2 \
                    * np.exp(k_p * xi_p / 4) \
                    * np.sqrt( \
                        sigma**2 * gamma / (2 * k_p * A) \
                            * (1 + gamma / k_p)**(1 + k_p / gamma)
                        )
    
    @staticmethod
    def compute_spread_unitary(gamma: float, A: float, k: float,sigma: float, xi: float) -> float:
        """
        Args:
            gamma: risk adversion 
            A: shape component for intensity 
            k: scale component for intensity 
            sigma: volatility of dS
            xi: 
            Note that all parameters A, k, sigma must be in the same units.
        Returns:
            spread: distance between optimal bid and ask quotes (bid_delta + ask_delta)
        """
        return 2/gamma*np.log(1+gamma/k)+xi+np.exp(k/4*xi)*np.sqrt(sigma**2*gamma/2/k/A*(1+gamma/k)**(1+k/gamma))

    @classmethod
    def compute_spread(cls, gamma: float, A: float, k: float,sigma: float, xi: float, base_qty: float) -> float:
        return cls.compute_spread_unitary(gamma, A, k / base_qty, sigma * base_qty, xi * base_qty) / base_qty