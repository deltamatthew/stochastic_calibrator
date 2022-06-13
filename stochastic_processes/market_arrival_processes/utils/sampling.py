import numpy as np
import pandas as pd
from itertools import repeat

def estimate_lambda_series(empirical_deltas: pd.Series, delta: float, dT: float) -> float:
    """
    Function to estimate lambda using empirical deltas

    Args:
        empirical_delta: empirical distances away from reference prices (all deltas should be > 0)
        delta: distance away from the reference price
        dT: trade count sampling frequency (measured in s)

    Returns:
        lambda_hat: estimated lambda
    """
    sum_lambdas_hat = (empirical_deltas > delta).groupby(pd.Grouper(freq = f'{dT}S')).sum()
    lambdas_hat = sum_lambdas_hat.mean() / dT
    return lambdas_hat

def sample_lambda_series(empirical_deltas: pd.Series, delta: float, dT:float):
    sum_lambdas_hat = (empirical_deltas > delta).groupby(pd.Grouper(freq = f'{dT}S')).sum()
    return sum_lambdas_hat / dT

def sample_lambda_df(empirical_deltas: pd.Series, deltas: pd.Series, dT:float):
    lambdas_hat = pd.concat(list(map(sample_lambda_series, repeat(empirical_deltas), deltas, repeat(dT))), axis = 1)
    lambdas_hat.columns = deltas
    return lambdas_hat
