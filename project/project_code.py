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

SFC_PRESS = 1000   # hPa * 100 = Pa
TOA_PRESS = 1     # hPa * 100 = Pa
DP = 50          # hPa * 100 = Pa
#LEVELS = np.linspace(SFC_PRESS,TOA_PRESS,30)
LEVELS = np.arange(SFC_PRESS,TOA_PRESS,-1*DP)


def bold_axes(ax):
    """
    Sets matplotlib axes linewidths to 2, making them
    bold and more attractive

    Parameters
    -----------
    ax : mpl.Axes
        axes to be bolded
    """
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2)

    # increase tick width
    ax.tick_params(width=2)
    
    for label in ax.get_xticklabels():
        label.set_weight('bold')

    for label in ax.get_yticklabels():
        label.set_weight('bold')

### Question 1 - Build an N Layer atmosphere solver that accounts for a variable epsilon in each layer or can take the original code

def parameterize_emiss(lev,e1=0.1,e2=0.2,e3=0.3):
    if lev <= 400:
        return e1
    elif 400 < lev <= 800:
        return e2
    else:
        return e3

def sw_ozone(levs, strat=250):
    """
    Function to determine the shortwave absorption of ozone.
    At the Top of the Atmosphere (TOA), 5% of shortwave radiation is absorbed.
    The absorption halves at each consecutive level to the tropopause (25000 Pa).
    After the tropopause, no absorption occurs (multiplier = 1).
    
    Parameters:
    levs : array-like
        Array of pressure levels in Pascals (Pa), with the first entry being the TOA (highest pressure).
    strat : int, optional
        Pressure level corresponding to the tropopause (default is 25000 Pa).
    
    Returns:
    multipliers : numpy.ndarray
        Array of multipliers for each pressure level.
    """
    multipliers = np.ones_like(levs, dtype=float)
    
    start_multiplier = 0.05
    multipliers[-1] = start_multiplier 
    
    for i in range(len(levs)-2, -1, -1):  # Start from second to last level and move upward
        if levs[i] <= strat:  # Above or at the tropopause
            multipliers[i] = (0.5 *  multipliers[i + 1])  # Halve the multiplier from the level above
        else:  # Below the tropopause
            multipliers[i] = 1  # No absorption below tropopause
        print(f'multiplier: {multipliers[i]}, pressure: {levs[i]}')
    
    return multipliers
    
    
def n_layer_re(levs, epsilon, S0=1350, albedo=0.33, debug=False, nuke=False, e1=0.001, e2=0.002, e3 = 0.003):
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
    ### number of layers - type checking ###
    if isinstance(levs,(int,float)):
        # if a set number of layers are supplied, just set N to that
        N = levs
    else:
        # NOTE: should fix the type check here to be safer!
        N = len(levs) - 1

    ### create matrices to solve energy balance ###
    # coefficient matrix
    A = np.zeros([N+1, N+1])
    # flux matrix
    b = np.zeros(N+1)

    # initialise the temperature array to isothermal atmosphere to start
    #temps = np.full_like(b,init_temp)
    #print(epsilon)

    ### calculate emissivities ###
    if isinstance(epsilon,(int,float)):
        # if emissivity is constant
        if debug:
            print(f'Constant epsilon of {epsilon} used at all layers')
        # set all layers to emissivity of constant epsilon
        epsilon = np.full_like(b,epsilon)
        # change surface emissivity to 1
        epsilon[0] = 1.

    elif isinstance(epsilon,(list,np.ndarray)):
        # if a list of epsilons are already specified, add 1 at surface
        print(epsilon)
        epsilon = np.array(epsilon)
        epsilon[0] = 1.0
    else:
        epsilon = np.full_like(b,0)
        for idx, lev in enumerate(levs[1:]):
            epsilon[idx+1] = parameterize_emiss(lev)
        epsilon[0] = 1.0
            

    # if nuclear winter, start absorbing ozone
    if nuke:
        
        insolation = -S0/4 
        multipliers = sw_ozone(levs) # array of what to multiply the insolation by
        insolation_scaled = insolation # start with the full insolation beam
        strat = 250

        for idx, lev in enumerate(levs):
            if debug:
                print(f'multiplier at {lev} : {multipliers[idx]}')
                print(f'insolation at {lev} : {insolation_scaled}')
            if lev <= strat:
                b[idx] = insolation_scaled * multipliers[idx]  # Store the scaled insolation at this level
                # Update the insolation for the next level, applying the multiplier
                insolation_scaled = insolation_scaled - (insolation_scaled * multipliers[idx])
                print(f'insolation at {lev} = {insolation_scaled}')  
            else:
                b[idx] = 0  # Set to 0 for levels above the tropopause
        
        b[0] = insolation_scaled * (1-albedo)
        print(f'b : {b}')



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
def atmospheric_profile(temps_array,params,ax=None,title='Vertical Atmospheric Profile',levels=LEVELS,c=IBM_PURPLE,ref=None):
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
    
    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(12,8))
    
    # now, plot vertical profile of atmospheric temperature.
    # we assume position 0 is the surface, and the altitude increases with layer to TOA
    if levels is not None:
        ax.plot(temps_array,levels,c=c,lw=3,label='Modelled Temperature')
        if ref is not None:
            ax.plot(ref,levels,c=IBM_PINK,ls='--',lw=3,label='Modelled Temperature')
        ax.set_yscale('log')
        ax.invert_yaxis()
        
    else:
        ax.plot(temps_array,range(len(temps_array)),c=c,lw=3,label='Modelled Temperature')

    # now format the axes and set labels
    bold_axes(ax) # make axes bold and pretty
    ax.set_xlabel(r'Temperature $(K)$', fontsize=14, fontweight='bold')
    ax.set_ylabel('Altitude (Layers Above Surface)', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', loc='left')
    # unpack the parameter dictionary and add to figure
    param_text = [f"{key}={val}" for key,val in params.items()]
    ax.set_title(f'{" ".join(str(i) for i in param_text)}',fontsize=12,fontweight='bold',loc='right')

    return ax



### Question 2 - Write a suite of unit tests to ensure that the solver can solve for epsilon at varying levels of complexity
# Test 1: repeat tests from original lab
# Test 2: 3 layer variable epsilon
# Test 3: 4 layer variable epsilon
# note that the function should return the A matrix to ensure that the
# unit test is thorough and all coefficients are being solved for correctly

### Question 3 - Lets add a parameterization for emissivity:  e <= 400mb = 0.1, 400 < p < 800 mb = 0.2, p > 800 e =0.3
# Compare this to an average surface temperature of 280K - is this too hot? 

### Question 4 - parameterize for ozone - key absorber in shortwave. compare the two profiles. why is this the case

### Question 5 - vary the tropopause height - what happens to the surface temperature?

temps, parameters, a = n_layer_re(LEVELS,'a',nuke=False,debug=0,S0=1361,albedo=0.33)
temps2, parameters2, a2 = n_layer_re(LEVELS,'a',nuke=True,debug=0,S0=1361,albedo=0.33)
ax = atmospheric_profile(temps,parameters,levels=LEVELS,ref=temps2)

