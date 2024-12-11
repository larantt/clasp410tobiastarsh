import os
from pathlib import Path
import unittest
import numpy as np
import matplotlib.pyplot as plt
import constants as c

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

# WELL-MIXED CONCENTRATIONS
CO2 = 380 * 1e6     # ppm to mol/mol
CH4 = 1.72 * 1e6    # ppm to mol/mol
N2O = (280.0 / 1e6 ) * 1e6  # ppt/1e6 = ppm

# SUPPLIED CONCENTRATION PROFILES
H2O = c.manabe_wv
O3 = c.rose_o3

# PROPERTIES FOR EACH GAS
MM_GAS = {           # molar masses
    'CO2' : 44.009,  # g/mol
    'CH4' : 16.040,  # g/mol
    'N2O' : 44.013,  # g/mol
    'O3'  : 48.000,  # g/mol
    'H2O' : 18.015,  # g/mol
    'air' : 28.965   # g/mol
}


# absorption cross sections estimated from Petty figure 9.10 &
# bands correspond to 
K_SPEC = {               # absorbtion cross sections in **most important** bands
        'CO2' : 1e-23,  # m^2/mol (15 micron band)
        'CH4' : 1e-21,  # m^2/mol
        'N2O' : 3e-21,  # m^2/mol
        'O3'  : 1e-19,  # m^2/mol
        'H2O' : 1e-21   # m^2/mol
    }

# LAYER MODEL SETUP
N = len(H2O)
SFC_PRESS = 1000 * 100   # hPa * 100 = Pa
TOA_PRESS = 10 * 100     # hPa * 100 = Pa
DP = 50 * 100            # hPa * 100 = Pa
# LEVELS = np.linspace(SFC_PRESS,TOA_PRESS,N)
# LEVELS = np.arange(SFC_PRESS,TOA_PRESS,DP)
LEVELS = c.rose_levs * 100 # if using the CESM levels


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

def avg_abs_cross(p,conc_profs,spectra,molar_mass,temp=None,debug=0):
    """
    Function solves for the absorbtion cross section of a given
    layer N, used to calculate the emissivity of the layer later
    in the code.

    This is the sum of all of the specific absorbtion cross sections
    multiplied by the concentration of the gas constituent at that
    atmospheric level.

    Parameters
    ----------
    p : float
        pressure level to solve for (Pa)
    conc_prof : dict(np.array)
        concentration of each gas constituent at each pressure level (/mol)
    spectra : dict(np.array, int)
        if array, integrate to get to the band-averaged absorbtion cross section (microns)
        if int, assume band averaged absorbtion cross section is prescribed (microns)
    molar_mass : dict
        contains the molar mass of each gas constituent to determine the mass mixing ratio
        from the volumetric mixing ratio (g/mol)
    temp : float
        temperature (if using temperature dependent gas constituent (e..g WV)) (K)
    debug : int, default 0
        verbosity of debugging

    Returns
    -------
    k : float
        absorption cross section at the layer (microns/mol)
    """
    # NOTE: add handling for array spectra (extra loop)
    # loop over concentration profiles to find the layer total 
    k = 0
    for gas,profile in conc_profs.items():
        # check if water vapor, spectra etc.
        if temp:
            # unless i can figure out a way to add C-C scaling, probably shld change "temp" to "wv"
            # per Manabe model - use a specific humidity profile bc thats easier (account for C-C later on)
            if debug:
                print(f'getting temperature dependent profile for {gas} - assuming mass mixing ratio provided')
            # NOTE: CHECK UNITS HERE! SHOULD BE IN KG/KG not G/KG LIKE STANDARD Q!!!! (either solve here or when initialising)
            # probably solve here for a fun and mean units trick :0
            q_p = profile[p]    # specific humidity is already a mass mixing ratio
        else:
            # account for ideal gas law and convert to mass mixing ratio
            q_p = profile[p]  * (molar_mass['air']/molar_mass[gas]) # g/g == kg/kg, move forward
        
        # get the absorbtivity cross section for that constituent
        k_p = spectra[gas]
        #print(q_p )
        # sum for the absorption cross section
        k += k_p * q_p  # m**2/kg
    
    return k

def calc_emissivity(k,dp,g=9.81,temp=None):
    """
    Function calculates the emissivity of a layer N
    by calculating the optical depth of the (very small) layer
    and then using this to determine the emissivity based on
    Beer's law. Note that this relies on a discretised estimation
    of optical depth.

    Parameters
    ----------
    k : float
        layer average absorption cross section (microns/mol)
    dp : float
        change in pressure over level (Pa), note that for now
        we are working w constant dp
    g : float, default 9.81
        acceleration due to gravity (ms-2)
    temp : float, default None
        temperature (if accounted for in optical depth calculation) (K)
    
    Return
    ------
    epsilon : float
        emissivity calculated for the layer (unitless)
    """
    # NOTE: add the check for temperature dependence
    # calculate the optical depth of the layer - unit area column so mass = p * 1/g 
    tau = (-1/g) * k * dp  # unitless
    print(k)
    # solve for the emissivity
    epsilon = 1 - np.exp(-1*tau)  # unitless

    return epsilon

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


def n_layer_re(levs, epsilon, S0=1350, albedo=0.33, debug=False, nuke=False, init_temp=50):
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
        N = len(levs)

    ### create matrices to solve energy balance ###
    # coefficient matrix
    A = np.zeros([N+1, N+1])
    # flux matrix
    b = np.zeros(N+1)

    # initialise the temperature array to isothermal atmosphere to start
    temps = np.full_like(b,init_temp)

    ### calculate emissivities ###
    if isinstance(epsilon,(int,float)):
        # if emissivity is constant
        if debug:
            print(f'Constant epsilon of {epsilon} used at all layers')
        # set all layers to emissivity of constant epsilon
        epsilon = np.full_like(b,epsilon)
        # change surface emissivity to 1
        epsilon[0] = 1.

    elif isinstance(epsilon,list):
        # if a list of epsilons are already specified, add 1 at surface
        epsilon = np.array(epsilon)
        epsilon = np.insert(epsilon,0,1)

    else:
        epsilon = np.zeros(N+1)
        # if no epsilon is specified, calculate emissivities from rad tran equations
        # initialise gas concentration profiles in dictionary
        gas_concs = {
            'CO2' : np.full(N,CO2),
            'H2O' : H2O,
            'CH4'  : np.full(N,CH4)
        }
    
        # loop over the pressure differences
        for idx,pres in enumerate(levs):
            if idx == 0:
                # first layer, emissivity is 1
                epsilon[idx] = 1.0
            else:
                # get pressure at that level 
                dp = levs[idx-1] - pres
                k = avg_abs_cross(idx,gas_concs,K_SPEC,MM_GAS)
                #print(k)
                # calculate emissivity
                eps = calc_emissivity(k,dp)
                #print(eps)
                epsilon[idx] = eps
            


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
temps, parameters, a = n_layer_atmos(5,1)
temps, parameters, a = n_layer_re(LEVELS,'a')
atmospheric_profile(temps,parameters)