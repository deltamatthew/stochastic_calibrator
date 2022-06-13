from numba import   jit, njit,\
                    float64, int64, b1
from .helpers import pd2np2pd_decorator
import numpy as np

@njit(int64(int64, int64))
def floor_ts(ts, freq_in_ms):
    return (ts // freq_in_ms) * freq_in_ms

@njit(b1(int64, int64, int64))
def check_if_new_bin(prev_ts, curr_ts, freq_in_ms):
    return curr_ts > prev_ts + freq_in_ms

# ema classes
@pd2np2pd_decorator
def ewma_for_df(arr, alpha):
    """
        fast ema function for df
    """
    return _ewma_2d(arr, alpha)

@jit((float64[:,:], float64), nopython=True, nogil=True)
def _ewma_2d(arr_in, alpha):
    r"""Exponentialy weighted moving average specified by a decay ``float``
    to provide better adjustments for small windows via:

        y[t] = (x[t] + (1-a)*x[t-1] + (1-a)^2*x[t-2] + ... + (1-a)^n*x[t-n]) /
               (1 + (1-a) + (1-a)^2 + ... + (1-a)^n).

    Parameters
    ----------
    arr_in : np.ndarray, float64
        A single dimenisional numpy array
    alpha: float64
        The decay factor

    Returns
    -------
    np.ndarray
        The EWMA vector, same length / shape as ``arr_in``

    Examples
    --------
    >>> import pandas as pd
    >>> a = np.arange(5, dtype=float)
    >>> exp = pd.DataFrame(a).ewm(span=10, adjust=True).mean()
    >>> np.array_equal(_ewma_infinite_hist(a, alpha = 2/(10 + 1)), exp.values.ravel())
    True
    """
    n = arr_in.shape
    ewma = np.empty(n, dtype=np.float64)
    w = 1
    ewma_old = arr_in[0]
    ewma[0] = ewma_old
    for i in range(1, n[0]):
        w += (1-alpha)**i
        ewma_old = ewma_old*(1-alpha) + arr_in[i]
        ewma[i] = ewma_old / w
    return ewma


@jit((float64[:], float64), nopython=True, nogil=True)
def _ewma(arr_in, alpha):
    r"""Exponentialy weighted moving average specified by a decay ``window``
    to provide better adjustments for small windows via:

        y[t] = (x[t] + (1-a)*x[t-1] + (1-a)^2*x[t-2] + ... + (1-a)^n*x[t-n]) /
               (1 + (1-a) + (1-a)^2 + ... + (1-a)^n).

    Parameters
    ----------
    arr_in : np.ndarray, float64
        A single dimenisional numpy array
    alpha: float64
        The decay factor

    Returns
    -------
    np.ndarray
        The EWMA vector, same length / shape as ``arr_in``

    Examples
    --------
    >>> import pandas as pd
    >>> a = np.arange(5, dtype=float)
    >>> exp = pd.DataFrame(a).ewm(span=10, adjust=True).mean()
    >>> np.array_equal(_ewma_infinite_hist(a, 10), exp.values.ravel())
    True
    """
    n = arr_in.shape[0]
    ewma = np.empty(n, dtype=np.float64)
    w = 1
    ewma_old = arr_in[0]
    ewma[0] = ewma_old
    for i in range(1, n):
        w += (1-alpha)**i
        ewma_old = ewma_old*(1-alpha) + arr_in[i]
        ewma[i] = ewma_old / w
    return ewma


@jit((float64[:,:], float64), nopython=True, nogil=True)
def _ewma_infinite_hist_2d(arr_in, alpha):
    r"""Exponentialy weighted moving average specified by a decay ``alpha``
    assuming infinite history via the recursive form:

        (2) (i)  y[0] = x[0]; and
            (ii) y[t] = a*x[t] + (1-a)*y[t-1] for t>0.

    This method is less accurate that ``_ewma`` but
    much faster:

        In [1]: import numpy as np, bars
           ...: arr = np.random.random(100000)
           ...: %timeit bars._ewma(arr, 10)
           ...: %timeit bars._ewma_infinite_hist(arr, 10)
        3.74 ms ± 60.2 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
        262 µs ± 1.54 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

    Parameters
    ----------
    arr_in : np.ndarray, float64
        A single dimenisional numpy array
    alpha: float64
        The decay factor

    Returns
    -------
    np.ndarray
        The EWMA vector, same length / shape as ``arr_in``

    Examples
    --------
    >>> import pandas as pd
    >>> a = np.arange(5, dtype=float)
    >>> exp = pd.DataFrame(a).ewm(span=10, adjust=False).mean()
    >>> np.array_equal(_ewma_infinite_hist(a, 10), exp.values.ravel())
    True
    """
    n = arr_in.shape
    ewma = np.empty(n, dtype=np.float64)
    ewma[0] = arr_in[0]
    for i in range(1, n[0]):
        ewma[i] = arr_in[i] * alpha + ewma[i-1] * (1 - alpha)
    return ewma

@jit((float64[:], float64), nopython=True, nogil=True)
def _ewma_infinite_hist(arr_in, alpha):
    r"""Exponentialy weighted moving average specified by a decay ``alpha``
    assuming infinite history via the recursive form:

        (2) (i)  y[0] = x[0]; and
            (ii) y[t] = a*x[t] + (1-a)*y[t-1] for t>0.

    This method is less accurate that ``_ewma`` but
    much faster:

        In [1]: import numpy as np, bars
           ...: arr = np.random.random(100000)
           ...: %timeit bars._ewma(arr, 10)
           ...: %timeit bars._ewma_infinite_hist(arr, 10)
        3.74 ms ± 60.2 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
        262 µs ± 1.54 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

    Parameters
    ----------
    arr_in : np.ndarray, float64
        A single dimenisional numpy array
    alpha: float64
        The decay factor

    Returns
    -------
    np.ndarray
        The EWMA vector, same length / shape as ``arr_in``

    Examples
    --------
    >>> import pandas as pd
    >>> a = np.arange(5, dtype=float)
    >>> exp = pd.DataFrame(a).ewm(span=10, adjust=False).mean()
    >>> np.array_equal(_ewma_infinite_hist(a, 10), exp.values.ravel())
    True
    """
    n = arr_in.shape[0]
    ewma = np.empty(n, dtype=float64)
    ewma[0] = arr_in[0]
    for i in range(1, n):
        ewma[i] = arr_in[i] * alpha + ewma[i-1] * (1 - alpha)
    return ewma

