import numba as nb
import numpy as np

@nb.njit(nb.float64(nb.float64, nb.float64, nb.float64, nb.float64, nb.float64, nb.float64))
def compute_bid_delta_vanilla(gamma:float, A:float, k:float, sigma:float, q:float, base_qty:float) -> float:
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

@nb.njit(nb.float64(nb.float64, nb.float64, nb.float64, nb.float64, nb.float64, nb.float64))
def compute_ask_delta_vanilla(gamma:float, A:float, k:float, sigma:float, q:float, base_qty: float) -> float:

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

@nb.njit(nb.float64(nb.float64, nb.float64, nb.float64, nb.float64, nb.float64, nb.float64, nb.float64))
def compute_ask_delta_market_impact(gamma: float, A: float, k: float,sigma: float, xi: float, q: float, base_qty: float) -> float:
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

@nb.njit(nb.float64(nb.float64, nb.float64, nb.float64, nb.float64, nb.float64, nb.float64, nb.float64))
def compute_bid_delta_market_impact(gamma: float, A: float, k: float,sigma: float, xi: float, q: float, base_qty: float) -> float:
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