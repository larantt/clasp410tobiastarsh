#!/usr/bin/env python3
"""
lab03.py
"""
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

# define our main solver for the lab
def n_layer_atmos(N, epsilon, S0=1350, albedo=0.33, debug=False, nuke=False):
    '''
    Solve the n-layer atmosphere problem and return temperature at each layer.

    Parameters
    ----------
    N : int
        Set the number of layers.
    epsilon : float, default=1.0
        Set the emissivity of the atmospheric layers.
    albedo : float, default=0.33
        Set the planetary albedo from 0 to 1.
    S0 : float, default=1350
        Set the incoming solar shortwave flux in Watts/m^2.
    debug : boolean, default=False/0
        Turn on debug output. Integer determines level of 
        verbosity of the output. 1 will only return the coefficient
        matrix for use in unit tests, 2 will print the A matrix,
        greater than 2 will print all information.

    Returns
    -------
    temps : Numpy array of size N+1
        Array of temperatures from the Earth's surface (element 0) through
        each of the N atmospheric layers, from lowest to highest.
    '''

    # Create matrices:
    A = np.zeros([N+1, N+1])

    b = np.zeros(N+1)
    # if nuclear winter, set the top of the b-matrix to S flux, atmosphere opaque to SW
    if nuke:
        b[-1] = -S0/4 * (1-albedo)
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
                A[i, j] = -1*(i > 0) - 1
                # print full debugging if verbose
                if debug > 2:
                    print(f"Result: A[i,j]={A[i,j]}")
            else:
                # use the emissivity pattern in the matrix
                m = np.abs(j-i) - 1
                A[i, j] = epsilon * (1-epsilon)**m
    # divide out the additional factor of epsilon at surface because epsilon = 1
    A[0, 1:] /= epsilon

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

# define our suite of unit tests for the lab
class TestNlayeratmos(unittest.TestCase):
    """
    Class contains unit tests for the verification step in the 
    lab methodology. While this would normally be run as its 
    own unit, this is run in the question 2 function below for ease 
    of use and grading
    """
    def test_2layer_halfepsilon(self):
        """
        First unit test, solves a 2 layer atmosphere
        with 0.5 for emissivity. Taken from the in class
        example lecture video.
        """
        # define the observations array for comparison
        obs_temps = np.array([285.5, 251.3, 227.1])
        obs_A = np.array([[-1.,1.,0.5],
                          [0.5,-2.,0.5],
                          [0.25,0.5,-2.]])
        # confirm the coefficients and the temperatures are equal to 1 decimal place
        np.testing.assert_array_equal(np.round(n_layer_atmos(2, 0.5)[0],1), obs_temps)
        np.testing.assert_array_equal(n_layer_atmos(2, 0.5)[-1], obs_A)

    def test_3layer_halfepsilon(self):
        """
        Second unit test, solves a 3 layer atmosphere 
        with 0.5 for emissivity. Taken from the in class
        example lecture video.
        """
        # define the observations array for comparison
        obs_temps = np.array([298.8, 270.0, 251.3, 227.1])
        obs_A = np.array([[-1.,1.,0.5,0.25],
                          [0.5,-2.,0.5,0.25],
                          [0.25,0.5,-2.,0.5],
                          [0.125,0.25,0.5,-2]])
        # confirm the coefficients and the temperatures are equal to 1 decimal place
        np.testing.assert_array_equal(np.round(n_layer_atmos(3, 0.5)[0],1), obs_temps)
        np.testing.assert_array_equal(n_layer_atmos(3, 0.5)[-1], obs_A)

    def test_2layer_thirdepsilon(self):
        """
        Third unit test, solves a 3 layer atmosphere with 1/3 for
        emissivity. Taken from the in class example lecture video.
        """
        # define the observations array for comparison
        obs_temps = np.array([273.4, 237.7, 221.2])
        obs_A = np.array([[-1.,1.,2/3],
                          [1/3,-2.,1/3],
                          [2/9,1/3,-2.]])
        # confirm the coefficients and the temperatures are equal to 1 decimal place
        np.testing.assert_array_equal(np.round(n_layer_atmos(2, 1/3)[0],1), obs_temps)
        # here due to precision errors in rounding 1/3 use high tolerance all close
        np.testing.assert_allclose(n_layer_atmos(2, 1/3)[-1], obs_A, rtol=1E7)

    def test_1layer_1epsilon(self):
        """
        Third unit test, solves a 1 layer atmosphere with 1 for
        emissivity. Taken from the in class example.
        """
        # define the observations array for comparison
        obs_temps = np.array([298.8,251.3])
        obs_A = np.array([[-1,1],
                          [1,-2]])
        # confirm the coefficients and the temperatures are equal to 1 decimal place
        np.testing.assert_array_equal(np.round(n_layer_atmos(1, 1)[0],1), obs_temps)
        np.testing.assert_array_equal(n_layer_atmos(1, 1)[-1], obs_A)


# function for atmospheric profile
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

def emissivity_profile(sfc_temps_array,xaxis_array,params,vary=r'Emissivity $(\epsilon)$',title=None):
    """
    Function creates a plot of temperature vs atmospheric emissivity

    Parameters
    ----------
    sfc_temps_array : np.ndarray
        array of temperatures at each layer, with position 0
        being the surface
    xaxis_array : np.ndarray
        array of parameter being varied for each temperature
    params : dict
        parameters used in the N-Layer atmosphere model
    vary : raw string
        parameter in the equation being varied, used for
        annotation. Defaults to emissivity. Include units if
        desired in annotation.
    title : string
        title if supplied, otherwise constructed using vary

    Returns
    -------
    fig, ax : mpl.figure, axis
        figure and axis created if not provided to the function
    """

    fig, ax = plt.subplots(1,1,figsize=(12,8))
    
    # we assume position 0 is the surface, and the altitude increases with layer to TOA
    ax.plot(xaxis_array,sfc_temps_array,c=IBM_PURPLE,lw=3)

    if not title:
        title = f'Surface Temperature {r"$(K)$"} vs {vary}'

    # now format the axes and set labels
    bold_axes(ax) # make axes bold and pretty
    ax.set_ylabel(r'Surface Temperature $(K)$', fontsize=14, fontweight='bold')
    ax.set_xlabel(vary, fontsize=14, fontweight='bold')
    ax.set_title(f'Surface Temperature {r"$(K)$"} vs {vary}', fontsize=16, fontweight='bold', loc='left')
    # unpack the parameter dictionary and add to figure
    param_text = [f"{key}={val}" for key,val in params.items()]
    ax.set_title(f'{" ".join(str(i) for i in param_text)}',fontsize=12,fontweight='bold',loc='right')

    return fig, ax

# Question 2 - validate the model against the website/examples from class
def question_two():
    """
    Function creates a plot to verify our model for a 5 layer
    atmosphere, with emissivity of 0.5 and albedo of 0.1
    which is more complex than I could be bothered to 
    solve the coefficients for and include in the unit tests. 

    It also effectively acts as a test for the atmospheric_profile
    function, and I really wrote this to debug the plotting function
    on top of the unit tests which I had written first.

    The temperatures for each layer were derived from:
    https://singh.sci.monash.edu/Nlayer.shtml

    """
    
    # then solve for a five layer atmosphere from the website in docstring
    expr2_obs = np.array([345.7,321.7,307.4,290.7,270.5,244.5])
    expr2_temps, expr2_params, _ = n_layer_atmos(5,0.5,albedo=0.1)
    print('Testing atmosphere with 5 layers, emissivity of 0.5, albedo of 0.1:')
    print(f'Website Surface Temperature: {expr2_obs[0]}K')
    print(f'Modelled Surface Temperature {np.round(expr2_temps[0],1)}K')

    # plot results of experiment
    fig1, ax1 = atmospheric_profile(expr2_temps,expr2_params,
                                    'Comparison of Verification and Modelled Atmospheric Profiles')
    ax1.scatter(expr2_obs,range(len(expr2_obs)),marker='x',c='k',
                zorder=100,label='Verification Temperatures')
    ax1.legend(ncols=1,frameon=False,prop={"size" : 12, "weight" : 'bold'})
    
    # save the figure to disk
    fig1.savefig(f'{OUTPATH}/q2_verification.png',dpi=300)



# Question 3 - How does the surface temperature of Earth depend on emissivity and the number of layers?.
def question_three():
    """
    Function conducts experiments and creates necessary plots to answer
    Question 3 in the lab.

    The first experiment varies the emissivity of the atmospheric layers and 
    creates a profile plot of emissivity with surface temperature for a 1 layer
    atmosphere. 

    The second experiment varies the number of atmospheric layers with a constant
    emissivity of 0.255 and creates vertical profile plot of temperature with
    altitude (number of layers in the atmosphere)

    """
    print('\nQ3: How Does the Surface Temperature of the Earth Depend on emissivity and # of layers')
    # first solve surface temperature vs emissivity for a one layer atmosphere w the same params
    expr1_epsilon = np.arange(0.05,1.05,0.05)
    # don't bother saving the parameter dictionaries or the first atmospheric layer
    expr1_temps = [n_layer_atmos(1,eps)[0][0] for eps in expr1_epsilon]
    # construct our own parameter dictionary without the epsilon as it is being varied
    expr1_param = {
        'N'           : 1,
        r'$S_{0}$'    : 1350,
        r'$\alpha$' : 0.33
    }
    # calculate, print and annotate the emissivity at 288K
    sfc_temps_1 = np.array(expr1_temps)
    # get the index (corresponds to number of layers) of closest temp to 288K
    idx_1 = (np.abs(sfc_temps_1 - 288)).argmin()
    print(f'Emissivity for a Surface Temperature of 288K: {expr1_epsilon[idx_1]})')
    
    # plot the surface temperature vs emissivity
    fig1, ax1 = emissivity_profile(expr1_temps,expr1_epsilon,expr1_param)
    # now plot a horizontal line at 288K to illustrate this on the plot
    ax1.axhline(288,ls='--',lw=3,c="#A1AEB1",label='288 K (Earth Surface Temperature)')
    ax1.legend(ncols=1,frameon=False,prop={"size" : 12, "weight" : 'bold'})
    
    fig1.savefig(f'{OUTPATH}/q3_emissivity.png',dpi=300)

    # next calculate surface temperature and vary N instead
    layers = np.arange(0,10,1)
    expr2_temps = [n_layer_atmos(N,0.255)[0] for N in layers] # don't bother saving the parameter dictionaries
    # get surface temperatures
    sfc_temps_2 = np.array([temp[0] for temp in expr2_temps])
    # get the index (corresponds to number of layers) of closest temp to 288K
    idx_2 = (np.abs(sfc_temps_2 - 288)).argmin()
    print(f'Number of layers to achieve 288K surface temperature: {idx_2} ({sfc_temps_2[idx_2]}K)')
    # give the temperature above if we are below the target temperature at nearest layer
    # this would be caused by a smaller distance from the desired sfc temp
    if sfc_temps_2[idx_2] < 288:
        print(f'                                                  {idx_2+1} ({sfc_temps_2[idx_2+1]}K)')

    # construct our own parameter dictionary without the epsilon as it is being varied
    earth_param = {
        'N'             : idx_2,
        r'$\epsilon$' : 0.255,
        r'$S_{0}$'      : 1350,
        r'$\alpha$'   : 0.33,
    }

    # plot altitude vs temperature
    fig2, ax2 = atmospheric_profile(expr2_temps[idx_2],earth_param,title='Vertical Atmospheric Profile of Earth')

    fig2.savefig(f'{OUTPATH}/q3_vertical.png',dpi=300)


# Question 4 - How many atmospheric layers do we expect on the planet Venus?
def question_four(s0=2600,alpha=0.33,sfc_temp=700,emis=1,planet='Venus'):
    """
    Function runs the experiments and creates the figures necessary
    to answer Question 4 in the lab manual. This seeks to understand how many atmospheric
    layers Venus has, given the incoming solar shortwave flux of 2600 W/m2, a surface
    temperature of 700K and an emissivity of 1.

    To do this, we vary the number of layers and plot the surface temperature against
    the number of atmospheric layers.

    Function is written such that it could be applied to any planet.
    """
    print('\nQ4: How Many Atmospheric Layers on Venus?')
    layers = np.arange(0,50,1)
    # don't bother saving the parameter dictionaries or the first atmospheric layer
    # assume albedo is the same as Earth (0.33)
    sfc_temps_arr = np.array([n_layer_atmos(N,emis,s0,alpha)[0][0] for N in layers])
    # construct our own parameter dictionary without the epsilon as it is being varied
    param =  {
        r'$\epsilon$' : emis,
        r'$S_{0}$'      : s0,
        r'$\alpha$'   : alpha
    }
    # get the index (corresponds to number of layers) of closest temp to 288K
    idx = (np.abs(sfc_temps_arr - sfc_temp)).argmin()
    print(f'Number of layers to achieve {sfc_temp}K surface temperature: {idx} ({sfc_temps_arr[idx]}K)')
    # give the temperature above if we are below the target temperature at nearest layer
    # this would be caused by a smaller distance from the desired sfc temp
    if sfc_temps_arr[idx] < sfc_temp:
        print(f'                                                      {idx+1} ({sfc_temps_arr[idx+1]}K)')

    # plot the surface temperature vs emissivity
    fig1, ax1 = emissivity_profile(sfc_temps_arr,layers,param,vary='Atmospheric Layers')
    # plot a dashed vertical line at the observed surface temperature of Venus
    ax1.axhline(sfc_temp,ls='--',lw=3,c="#A1AEB1",label=f'{sfc_temp}K ({planet} Surface Temperature)')
    ax1.legend(ncols=1,frameon=False,prop={"size" : 12, "weight" : 'bold'})
    
    fig1.savefig(f'{OUTPATH}/q4_{planet}.png',dpi=300)


# Question 5 - nuclear winter
def question_five():
    """
    Function runs the experiments and creates the figures necessary to 
    answer question five in the lab report. This question corresponds to a 
    nuclear winter scenario, where the entire solar flux is only absorbed by the 
    top of the atmosphere and the surface is opaque. The only difference is that
    the b matrix in n_layer_atmos is flipped for this scenario.

    Here we have a 5 layer atmosphere with an emissivity of 0.5 and s0
    of 1350W/m2. This produces a vertical profile of the atmosphere for this 
    scenario.
    """
    print('\nQ5: What would Earths Surface Temperature be under Nuclear Winter?')
    nuclear_temps, nuclear_params, _ = n_layer_atmos(5,0.5,nuke=True) # _ because don't save A matrix
    print(f'Surface Temperature: {nuclear_temps[0]}')
    # plot results of second experiment
    fig1, ax1 = atmospheric_profile(nuclear_temps,nuclear_params,
                                    'Atmospheric Vertical Profile for Nuclear Winter Scenario')
    
    # save the figure to disk
    fig1.savefig(f'{OUTPATH}/q5_nuclear.png',dpi=300)

def main():
    """
    main function executes all code necessary to reproduce
    the figures in the lab report. Will execute all code in order.
    """
    # create path for figures to be saved if not already existing
    global OUTPATH
    if not OUTPATH:
        # will create a figures directory in your current working directory
        OUTPATH = f'{os.getcwd()}/laratt_lab03_figures'

    Path(OUTPATH).mkdir(parents=True, exist_ok=True)

    question_two()
    question_three()
    question_four()
    question_five()

# run the main function when calling script
if __name__ == "__main__":
    # first run the unit tests to ensure everything works
    # do not exit the script
    print('testing n_layer_atmos solver')
    unittest.main(exit=False)
    # now execute the code for the class if everything works
    print('\nExecuting Lab Experiments')
    print('--------------------------')
    main()
