import os
from pathlib import Path
import unittest
import numpy as np
import matplotlib.pyplot as plt

# ------- #
# GLOBALS #
# ------- #
# make math text look nice (add later)
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams["mathtext.default"] = 'bf'

# set the Stefan-Boltzmann Constant
SIGMA = 5.67E-8
# define nice colorblind friendly plotting colors
# source: https://davidmathlogic.com/colorblind/#%23648FFF-%23785EF0-%23DC267F-%23FE6100-%23FFB000
IBM_BLUE = "#648FFF"
IBM_PURPLE = "#785EF0"
IBM_PINK = "#DC267F"
IBM_ORANGE = "#FE6100"
IBM_YELLOW = "#FFB000"
# set outpath - user should change to ensure figures save correctly or specify None, at which point
# the main() function will determine the outpath using the os module and create a figures directory
OUTPATH = "/Users/laratobias-tarsh/Documents/fa24/clasp410tobiastarsh/labs/lab03/figures"

### Question 1 - Build an N Layer atmosphere solver that accounts for a variable epsilon in each layer or can take the original code

def n_layer_atmos(N, epsilon, S0=1350, albedo=0.33, debug=False, nuke=False):
    '''
    Solve the n-layer atmosphere problem and return temperature at each layer.

    Parameters
    ----------
    N : int
        number of atmospheric layers
    epsilon : arrayLike or Float
        emissivity of each of atmospheric layers
        if float, will assume that all layers are that float
    albedo : float, default=0.33
        planetary albedo (ranges from 0-1)
    S0 : float, default=1350
        Set the incoming solar shortwave flux in Watts/m^2.
    debug : bool, default=False/0
        turn on debug output. Integer determines level of 
        verbosity of the output. 1 will only return the coefficient
        matrix for use in unit tests, 2 will print the A matrix,
        greater than 2 will print all information.
    nuke : bool, default=False
        nuclear winter scenario.

    Returns
    -------
    temps : np.ndarray
        array of temperatures from the Earth's surface (element 0) through
        each of the N atmospheric layers, from lowest to highest
    params : dict
        parameters used in the model, used for plotting
    A : np.ndarray
        coefficients matrix, used for debugging
    '''

    # Create matrices:
    A = np.zeros([N+1, N+1])

    b = np.zeros(N+1)

    if isinstance(epsilon,(int,float)):
        print('Constant epsilon')
        epsilon = np.full_like(b,epsilon)
        epsilon[0] = 1.

    else:
        # append 1 to the start of the list
        epsilon = np.array(epsilon)
        epsilon = np.insert(epsilon,0,1)

    # if nuclear winter, set the top of the b-matrix to S flux, atmosphere opaque to SW
    if nuke:
        b[-1] = -S0/4
    # otherwise, atmosphere is transparent to shortwave and is absorbed at ground
    else:
        b[0] = -S0/4 * (1-albedo)

    # only print at maximum debug level
    if debug > 2:
        print(f"Populating N+1 x N+1 matrix (N = {N})")

    # populate the A matrix step-by-step
    for i in range(N+1):
        for j in range(N+1):
            # print full debugging if verbose
            if debug > 2:
                print(f"Calculating point i={i}, j={j}")
            # set all diagonal elements to -2 except the surface (i == 0)
            if i == j:
                A[i, j] = (-1*(i > 0) - 1) 
                # print full debugging if verbose
                if debug > 2:
                    print(f"Result: A[i,j]={A[i,j]}")
            elif j < i:
                m = epsilon[i]
                for k in range(j + 1, i):
                    m *= (1 - epsilon[k])
                A[i, j] = m
            else:
                m = epsilon[i]
                for ix in range(i + 1, j):
                    m *= (1 - epsilon[ix])
                A[i, j] = m

    # print verification of the A matrix
    if debug:
        print(f'A coefficient matrix:\n{A}')

    # Get the inverse of A matrix.
    a_inv = np.linalg.inv(A)

    # Multiply a_inv by b to get fluxes.
    fluxes = np.matmul(a_inv, b)

    # calculate temperatures from fluxes using stefan-boltzmann
    temps = (fluxes/epsilon/SIGMA)**0.25
    temps[0] = (fluxes[0]/SIGMA)**0.25  # Flux at ground: epsilon=1.

    # finally create a parameters dictionary to be used in plotting
    params = {
        'N'             : N,
        r'$\epsilon$' : epsilon,
        r'$S_{0}$'      : S0,
        r'$\alpha$'   : albedo
    }
    
    # if running in debug, will also return the coefficients matrix
    # this allows the unit tests to be called effectively
    return temps, params, A

## quick plotting function for developing the code
def atmospheric_profile(temps_array,params,ax=None,title='Vertical Atmospheric Profile'):
    """
    Function plots an atmospheric profile with altitude

    Parameters
    ----------
    temps_array : np.ndarray
        array of temperatures at each layer, with position 0
        being the surface
    params : dict
        dictionary of parameters to show on plot
    title : string
        ax suptitle for the plot

    Returns
    -------
    fig, ax : mpl.figure, axis
        figure and axis created if not provided to the function
    """
    
    fig, ax = plt.subplots(1,1,figsize=(12,8))
    
    # now, plot vertical profile of atmospheric temperature.
    # we assume position 0 is the surface, and the altitude increases with layer to TOA
    ax.plot(temps_array,range(len(temps_array)),c=IBM_PURPLE,lw=3,label='Modelled Temperature')

    # now format the axes and set labels
    bold_axes(ax) # make axes bold and pretty
    ax.set_xlabel(r'Temperature $(K)$', fontsize=14, fontweight='bold')
    ax.set_ylabel('Altitude (Layers Above Surface)', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', loc='left')
    # unpack the parameter dictionary and add to figure
    param_text = [f"{key}={val}" for key,val in params.items()]
    ax.set_title(f'{" ".join(str(i) for i in param_text)}',fontsize=12,fontweight='bold',loc='right')

    return fig, ax

### Question 2 - Write a suite of unit tests to ensure that the solver can solve for epsilon at varying levels of complexity
# Test 1: repeat tests from original lab
# Test 2: 3 layer variable epsilon
# Test 3: 4 layer variable epsilon
# note that the function should return the A matrix to ensure that the
# unit test is thorough and all coefficients are being solved for correctly

### Question 3 - Kerry Emanuel and CESM can give us known atmospheric profiles of key greenhouse gasses
# create a more comprehensive model for the main greenhouse constituents by calculating the actual emissivity
# of each layer (30 layer atmos?)

### Question 4 - Write a function to adjust the incoming solar radiation based on angle and date
# write more unit tests, and have student reproduce the Hartmann figure

### Question 5 - Finally, lets enforce convective equilibrium in the troposphere
temps, parameters, a = n_layer_atmos(30,0.2,albedo=0.299)