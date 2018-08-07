import operator as op

import pandas as pd


def mapna(s: pd.Series, *args, **kwargs) -> pd.Series:
    """ Like pd.Series.map, but with na_action='ignore' by default. """
    return s.map(*args, na_action='ignore', **kwargs)


def standardize(x: pd.Series, center: bool=True, scale: bool=True, skipna: bool=True) -> pd.Series:
    """ Center and scale an array """
    if center is True:
        center = x.mean(skipna=skipna)

    if center is not False and center is not None:
        x_centered = x - center
    else:
        x_centered = x
    
    if scale is True:
        scale = x.std(skipna=skipna)

    if scale is not False and scale is not None:
        x_scaled = x / scale
    else:
        x_scaled = x

    return x_scaled
