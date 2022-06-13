from numba.experimental import jitclass
from numba import   jit, njit, \
                    int32, float32, int64, int8, float64, \
                    types, typed, \
                    prange
import numpy as np

@njit(
    [(float64[:], float64)]
    )
def _calculate_lambda_nb(empirical_deltas: float64[:], delta: float64) -> int64: 
    return np.sum(empirical_deltas > delta)

@njit([
        (float64[:], float64[:])
    ],
    parallel = True
)
def calculate_lambdas_nb(empirical_deltas: float64[:], deltas: float64[:]) -> float:
    """
    Function to 1-step estimate lambda using empirical deltas

    Args:
        empirical_delta: empirical distances away from reference prices (all deltas should be > 0)
        delta: distance away from the reference price

    Returns:
        sum_lambdas_hat: new trades count
    """
    n = deltas.shape[0]
    sum_lambdas_hat = np.zeros(n)
    for i in prange(n):
        sum_lambdas_hat[i] = _calculate_lambda_nb(empirical_deltas, deltas[i])
    return sum_lambdas_hat
