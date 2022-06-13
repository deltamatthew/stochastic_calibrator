from math import sqrt, log

cdef class BrownianDiffusionEWM:

    def __init__(self, float dT, float sigma20, float alpha):
        self.dT = dT
        self.sigma2 = sigma20
        self.alpha = alpha

    cpdef update_sigma(self, float dS):
        self.sigma2 = self.alpha * dS**2 / self.dT + (<float>1.0 - self.alpha) * self.sigma2

    cpdef float get_sigma(self,):
        return sqrt(self.sigma2)

    cdef float _compute_ask_delta(self, float gamma, float A, float k, float sigma, float q):
        """
        Args:
            gamma: risk adversion 
            A: shape component for intensity 
            k: scale component for intensity 
            sigma: volatility of dS
            q: running quantity
            Note that all parameters A, k, sigma must be in the same units.
        Returns:
            bid_delta: distance away from reference for quoting bid (ref_price - bid_delta)
        """
        return 1/gamma*log(1+gamma/k)+(2*q-1)/2*sqrt(sigma**2*gamma/2/k/A*(1+gamma/k)**(1+k/gamma))

    cdef float _compute_ask_delta_base_qty(self, float gamma, float A, float k, float sigma, float q, float base_qty):
        cdef float k_p = k / base_qty
        cdef float q_p = q / base_qty
        cdef float sigma_p = sigma * base_qty
        return self._compute_ask_delta(gamma, A, k_p, sigma_p, q_p) / base_qty

    cpdef float compute_ask_delta(self, float gamma, float A, float k, float sigma, float q, float base_qty):
        return self._compute_ask_delta_base_qty(gamma, A, k, sigma, q, base_qty)

    cdef float _compute_bid_delta(self, float gamma, float A, float k, float sigma, float q):
        """
        Args:
            gamma: risk adversion 
            A: shape component for intensity 
            k: scale component for intensity 
            sigma: volatility of dS
            q: running quantity
            Note that all parameters A, k, sigma must be in the same units.
        Returns:
            bid_delta: distance away from reference for quoting bid (ref_price - bid_delta)
        """
        return 1/gamma*log(1+gamma/k)+(2*q+1)/2*sqrt(sigma**2*gamma/2/k/A*(1+gamma/k)**(1+k/gamma))

    cdef float _compute_bid_delta_base_qty(self, float gamma, float A, float k, float sigma, float q, float base_qty):
        cdef float k_p = k / base_qty
        cdef float q_p = q / base_qty
        cdef float sigma_p = sigma * base_qty
        return self._compute_bid_delta(gamma, A, k_p, sigma_p, q_p) / base_qty

    cpdef float compute_bid_delta(self, float gamma, float A, float k, float sigma, float q, float base_qty):
        return self._compute_bid_delta_base_qty(gamma, A, k, sigma, q, base_qty)

    cdef float _compute_spread(self, float gamma, float A, float k, float sigma):
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
        return 2/gamma*log(1+gamma/k)+sqrt(sigma**2*gamma/2/k/A*(1+gamma/k)**(1+k/gamma))

    cpdef float compute_spread(self, float gamma, float A, float k, float sigma):
        return self._compute_spread(gamma, A, k, sigma)