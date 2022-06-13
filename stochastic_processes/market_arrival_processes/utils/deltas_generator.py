import numpy as np


def _generate_deltas(r, tick_size, ):
    MAX_SLIPPAGE = 0.05
    TARGET_DATA_POINT = 100
    i_max = np.round(r * MAX_SLIPPAGE / tick_size)
    deltas = tick_size * np.logspace(np.log(1), np.log(i_max), TARGET_DATA_POINT, base=np.exp(1))
    return deltas