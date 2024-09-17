#!/usr/bin/env python

"""
example01coffee.py

This file produces the solution to the coffee problem which is really
important to everyone.

To use this file ...

This file solves Newton's law of cooling for ...
"""

## IMPORTS ##
import numpy as np
import matplotlib.pyplot as plt

## CONSTANTS ##
k = 1/300               # coeff of cooling in units of 1/s
T_init, T_env = 90, 20  # initial and environmental temps in units of C

## FUNCTIONS ##
def solve_temp(time, k=k, T_init=T_init, T_env=T_env):
    """
    Given an array of times in seconds, caluclate the
    temperature using Newton's Law of Cooling.

    Parameters
    ----------
    time :

    Kwargs
    ------
    k :

    T_init :

    T_env :

    Returns
    -------
    """
    return  T_env + (T_init - T_env) * np.exp(-k*time)


def time_to_temp(T_target, k=k, T_init=T_init, T_env=T_env):
    """
    For a given 

    Parameters
    ----------
    T_target :

    Kwargs
    ------
    k :

    T_init :

    T_env :

    Returns
    -------
    """
    pass
