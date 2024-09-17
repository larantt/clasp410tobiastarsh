#!/usr/bin/env python3
"""
diffs.py

Explore numerical differencing with python
"""

import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-talk')

def fwd_diff(y,dx):
    """
    Function returns a forward differencing approximation of a 
    first derivative.

    Parameters
    ----------
    y : float
        data to take the first derivative of
    dx : float
        interval at which to approximate the first derivative

    Returns
    -------
    dydx : arrayLike
        numpy array containing the value of the first derivative
        at each point.
    """
    # create an array of zeros to store the data in
    # should the size of the input data
    dydx = np.zeros(y.size)

    # take the forward difference of the equation up to the final point
    dydx[:-1] = (y[1:] - y[:-1]) /dx
    # take the backward difference at the final point to avoid going out
    # of the data range
    dydx[-1] = (y[-1] - y[-2]) /dx

    return dydx


def back_diff(y,dx):
    """
    Function returns a forward differencing approximation of a 
    first derivative.

    Parameters
    ----------
    y : float
        data to take the first derivative of
    dx : float
        interval at which to approximate the first derivative

    Returns
    -------
    dydx : arrayLike
        numpy array containing the value of the first derivative
        at each point.
    """
    # create an array of zeros to store the data in
    # should the size of the input data
    dydx = np.zeros(y.size)

    # take the backward difference except for the first point
    #dydx[1:] = (y[:-1] - y[1:]) /dx
    dydx[1:] = (y[1:] - y[:-1]) / dx
    # take the forward difference at the final point to avoid going out
    # of the data range
    dydx[0] = (y[1] - y[0]) /dx

    return dydx

# set a delta x and length of the data
deltax = 0.1
x = np.arange(0, 4*np.pi, deltax) # 4 cycles of sine

fx = np.sin(x)      # transform to a sine wave
fxd1 = np.cos(x)    # analytic solution, first derivative of sine is cosine

# now plot on figure
fig, ax = plt.subplots(1,1,figsize=(12,8))

# plot the sine wave (original points)
ax.plot(x, fx, '.', alpha=.3, label=r'$f(x) = \sin(x)$',c='k')

# plot the analytic solution
ax.plot(x, fxd1, label=r'$f(x) = \frac{d\sin(x)}{dx}$',c='#648FFF')
# plot the forward differencing solution
ax.plot(x,fwd_diff(fx,deltax),'--',label='forward diff',c='#DC267F')
# plot the backward differencing solution
ax.plot(x,back_diff(fx,deltax),'--',label='backward diff',c='#FFB000')
ax.legend(loc='upper right')

ax.set_title(r'Forward and Backward Difference Approximations of $f(x) = \frac{d\sin(x)}{dx}$',loc='left')

plt.show()
