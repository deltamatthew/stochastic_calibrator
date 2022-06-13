import numpy as np
import pandas as pd
import os

def shift_values(arr, shift, fill_value = np.nan, drop = False):
    n = arr.shape[0]
    if not drop:
        result = np.ones(n) * fill_value
        if shift >= 0:
            result[shift:] = arr[:-shift]
        else:
            result[:-shift] = arr[shift:]
    else: 
        result = np.ones(n-shift)
        if shift >= 0:
            result = arr[:-shift]
        else:
            result = arr[shift:]
    return result

def pd2np2pd_decorator(func):
    def _inner(pd_obj, *args, **kwargs):
        if isinstance(pd_obj, (pd.Series, pd.DataFrame)):
            obj_class = pd.Series if isinstance(pd_obj, pd.Series) else pd.DataFrame
            index = pd_obj.index
            return obj_class(func(pd_obj.to_numpy(dtype = np.float64), *args, **kwargs), index = index)
        else: print('This is not an instance of pd object.')
    return _inner

def multi_pd2np2pd_decorator(func):
    """
        Description:
            wrapper for function of pandas object in this form:
            - func(s1: Union[pd.Series, pd.DataFrame], s2: Union[pd.Series, pd.DataFrame], **kwargs)->Union[pd.Series, pd.DataFrame]
            where the output class is specified by the input(if all pd inputs are pd.Series object then the output is pd.Series, otherwise it would be pd.DataFrame)
        Example:
        >>> @multi_pd2np2pd_decorator
        >>> def func(s1, s2, abc):
        >>>     print(s1, s2)
        >>>     return s1+abc
        >>> func(abs_delta, delta_series, abc = 3)
    """
    def _inner(*pd_objs, **kwargs):
        if all(isinstance(pd_obj, (pd.Series, pd.DataFrame)) for pd_obj in pd_objs):
            obj_class = pd.Series if all(isinstance(pd_obj, pd.Series) for pd_obj in pd_objs) else pd.DataFrame
            index = pd_objs[0].index
            return (
                obj_class(
                    func(
                        *(pd_obj.to_numpy(dtype = np.float64) for pd_obj in pd_objs),
                        **kwargs
                        ), 
                    index = index
                    ) 
            )
        else: print('This is not an instance of pd object.')
    return _inner

def calculate_alpha(length, required_p):
    return 1 - (1 - required_p)**(1 / length)


def calculate_lambda_np(empirical_deltas, deltas):
    return np.cumsum(np.histogram(empirical_deltas, bins = deltas)[0][::-1])[::-1]

from itertools import repeat

def resample_count_sum(empirical_deltas, delta, dT):
	return (empirical_deltas > delta).resample(pd.Grouper(freq = f'{dT}S')).sum()

def calculate_count_sum(empirical_deltas, delta, ):
	return (empirical_deltas > delta).sum()

def calculate_counts_sum(empirical_deltas, deltas, dT):
	return np.array(list(map(calculate_count_sum, repeat(empirical_deltas), deltas, repeat(dT)))).T
   
def resample_counts_sum(empirical_deltas, deltas, dT):
	return np.array(list(map(resample_count_sum, repeat(empirical_deltas), deltas, repeat(dT)))).T
