"""
project_code.py

This lab implements a solver for an N-Layer atmosphere model, using a 
system of linear equations and numpy's linear algebra capabilities. We
use this model to explore how parameterizing for emissivity and ozone affects
an atmospheric profile of Earth, allowing us to more effectively model a grey-gas
atmosphere. The parameterizations are modelled after a classic radiative equlibrium
profile from Manabe and Strickler.

This lab produces the following figures:
q3_profiles.png    : vertical profiles comparing a constant emissivity atmosphere to a 
                     parameterized emissivity atmosphere and standard lapse rates.
q4_profiles.png    : vertical profile comparing the impact of parameterizing for ozone in the stratosphere.
q4_trop.png        : vertical profile showing the impact of varying the height of the troposphere.
q5_profiles.png    : vertical profiles for average conditions in three locations.

USER INPUTS
-----------
OUTPATH : str
    absolute filepath to the directory where figures should be saved.
    this should be edited for the machine you are working on.

To run this file, execute the following command:
>>> python3 project_code.py

Ensure you have changed the OUTPATH global to your machine.

This should produce all figures in the lab report and save them to disk. It will also run
a short suite of unit tests to confirm the solver behaves as expected.
"""

import os
from pathlib import Path
import unittest
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
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
OUTPATH = "/Users/laratobias-tarsh/Documents/fa24/clasp410tobiastarsh/project/figures"

# define pressure levels to be used in lab
SFC_PRESS = 1000   # hPa * 100 = Pa
TOA_PRESS = 49     # hPa * 100 = Pa
DP = 50          # hPa * 100 = Pa
#LEVELS = np.linspace(SFC_PRESS,TOA_PRESS,30)
LEVELS = np.arange(SFC_PRESS,TOA_PRESS-1,-1*DP)

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

def format_profile(axis,xmin=200):
    """
    Quick helper function to format the axes of the temperature 
    profiles to reduce rewriting code. Sets them on an easily
    read log scale.

    Parameters
    ----------
    axis : mpl.axes
        axes to format
    """
    axis.set_yscale('log')

    # nicely label the axes for readability
    y_ticks = [1000, 850, 700, 500, 400, 300, 200, 50]
    axis.set_yticks(y_ticks)
    axis.set_yticklabels(y_ticks)

    axis.minorticks_on()
    # label axes
    axis.set_xlabel(r'Temperature $(K)$', fontsize=11, fontweight='bold')
    axis.set_ylabel('Pressure (hPa)', fontsize=11, fontweight='bold')
    # nice formatting
    bold_axes(axis)
    axis.invert_yaxis()
    axis.set_xlim(xmin)

### Question 1 - Build an N Layer atmosphere solver that accounts for a variable epsilon in each layer or can take the original code
def parameterize_emiss(lev,e1=0.1,e2=0.2,e3=0.3):
    """
    Parameterization for emissivity defined in question 3

    Parameters
    ----------
    lev : np.ndarray
        array of pressure levels over which the function should be solved
    e1, e2, e3 : float, default 0.1, 0.2, 0.3
        emissivity at a given level
    
    Returns
    -------
    e1, e2, e3 : float
        emissivity at a given level
    """
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
    levs : np.ndarray
        Array of pressure levels
    strat : int, default 250
        Pressure level corresponding to the tropopause
    
    Returns:
    multipliers : numpy.ndarray
        array of multipliers for each pressure level
    """
    multipliers = np.ones_like(levs, dtype=float)
    
    start_multiplier = 0.05
    multipliers[-1] = start_multiplier 
    
    for i in range(len(levs)-2, -1, -1):  # Start from second to last level and move upward
        if levs[i] <= strat:  # Above or at the tropopause
            multipliers[i] = (0.5 *  multipliers[i + 1])  # Halve the multiplier from the level above
        else:  # Below the tropopause
            multipliers[i] = 1  # No absorption below tropopause
    
    return multipliers
    
def n_layer_re(levs, epsilon=None, S=1350/4, albedo=0.33, debug=False, ozone=False, 
                e1=0.1, e2=0.2, e3 = 0.3, strat=250, enforce_blackbody=True):
    '''
    Solve the n-layer atmosphere problem and return temperature at each layer.

    Parameters
    ----------
    N : int
        number of atmospheric layers
    epsilon : arrayLike or Float
        emissivity of each of atmospheric layers
        if float, will assume that all layers are that float. 
        If none provided will parameterize.
    albedo : float, default=0.33
        planetary albedo (ranges from 0-1)
    S : float, default=1350/4
        Set the incoming solar shortwave flux in Watts/m^2 at TOA (!!!).
        Note that this is different from the previous lab so that we can 
        change insolation with latitude.
    debug : bool, default=False/0
        turn on debug output. Integer determines level of 
        verbosity of the output. 1 will only return the coefficient
        matrix for use in unit tests, 2 will print the A matrix,
        greater than 2 will print all information.
    ozone : bool, default=False
        should we apply a parameterization for ozone?
    e1, e2, e3: float, default 0.1, 0.2, 0.3
        emissivity values for parameterization in parameterize_emiss
    strat : float, default 250
        height of tropopause/start of stratosphere for use in 
        ozone parameterization
    enforce_blackbody : bool, default True
        should we enforce blackbody assmuption at the surface (aka emissivty = 1.0)
        only applies to a list of emissivities (e.g. if you wanted to test e = 0.95 at the surface)

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

    ### calculate emissivities ###
    if isinstance(epsilon,(int,float)):
        # if emissivity is constant
        if debug:
            print(f'Constant epsilon of {epsilon} used at all layers')
        # set all layers to emissivity of constant epsilon
        epsilon = np.full_like(b,epsilon)
        # change surface emissivity to 1
        if enforce_blackbody:
            if debug > 1:
                print('enforcing black body rules - surface emissivity set to 1.0')
            epsilon[0] = 1.0

    elif isinstance(epsilon,(list,np.ndarray)):
        if debug:
            print(f'Array of {epsilon}, len {len(epsilon)} passed\nEnsure you have included a surface emissivity in your array!')
        if len(epsilon) != N + 1:
            raise(ValueError(f'Array epsilon = {epsilon} has incorrect length {len(epsilon)}. Ensure array epsilon has length N+1'))
        # if a list of epsilons are already specified, add 1 at surface
        epsilon = np.array(epsilon)
        if enforce_blackbody:
            if debug > 1:
                print('enforcing black body rules - surface emissivity set to 1.0')
            epsilon[0] = 1.0
    else:
        if debug:
            print(f'parameterizing epsilon with e1={e1}, e2={e2}, e3={e3}')
        epsilon = np.full_like(b,0)
        for idx, lev in enumerate(levs[1:]):
            epsilon[idx+1] = parameterize_emiss(lev,e1,e2,e3)
        epsilon[0] = 1.0
            

    # if nuclear winter, start absorbing ozone
    if ozone:
        if debug > 1:
            print("Parameterizing for ozone")
        insolation = -S
        multipliers = sw_ozone(levs,strat=strat) # array of what to multiply the insolation by
        insolation_scaled = insolation # start with the full insolation beam

        for idx, lev in enumerate(levs):
            if debug > 2:
                print(f'multiplier at {lev} : {multipliers[idx]}')
                print(f'insolation at {lev} : {insolation_scaled}')
            if lev <= strat:
                b[idx] = insolation_scaled * multipliers[idx]  # Store the scaled insolation at this level
                # Update the insolation for the next level, applying the multiplier
                insolation_scaled = insolation_scaled - (insolation_scaled * multipliers[idx])
            else:
                b[idx] = 0  # Set to 0 for levels above the tropopause
        
        b[0] = insolation_scaled * (1-albedo)
        if debug > 2:
            print(f'b : {b}')

    # otherwise, atmosphere is transparent to shortwave and is absorbed at ground
    else:
        b[0] = -S * (1-albedo)

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
    # I actually don't use this in the lab but kept it to be consistent
    # with my Lab 3 code for reference.
    params = {
        'N'             : N,
        r'$\epsilon$' : epsilon,
        r'$S_{0}$'      : S,
        r'$\alpha$'   : albedo
    }
    
    # if running in debug, will also return the coefficients matrix
    # this allows the unit tests to be called effectively
    return temps, params, A

def lapse_rate(levels, sfc_temp, lapse_rate=9.8):
    """
    Function converts a constant lapse rate for a given surface
    temperature to a set lapse rate (DALR = 9.8 K/km, MALR = 6.0 K/km)
    by using the barometric formula.

    Parameters
    ----------
    levels : np.ndarray
        numpy array representing the vertical levels used for plotting
    sfc_temp : float
        temperature at the surface of an atmospheric profile to plot 
        the lapse rate for
    lapse_rate : float
        how much the temperature should decrease by with height

    Returns
    -------
    adiabat : np.ndarray
        array of temperatures representing the specific lapse rate
    """
    # barometric formula
    altitude = (1 - (levels / levels[0])**(1 / 5.256)) * 44330
    # adiabat for the level you are at
    adiabat = sfc_temp - lapse_rate * (altitude / 1000)
    return adiabat

### Question 2 - Write a suite of unit tests to ensure that the solver can solve for epsilon at varying levels of complexity
class TestNlayeratmos(unittest.TestCase):
    """
    Class contains unit tests for the verification step in the 
    lab methodology. While this would normally be run as its 
    own unit, this is run in the question 2 function below for ease 
    of use and grading.

    Adapted from https://www.geeksforgeeks.org/unit-testing-python-unittest/
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
        np.testing.assert_array_equal(np.round(n_layer_re(2, 0.5)[0],1), obs_temps)
        np.testing.assert_array_equal(n_layer_re(2, 0.5)[-1], obs_A)

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
        np.testing.assert_array_equal(np.round(n_layer_re(3, 0.5)[0],1), obs_temps)
        np.testing.assert_array_equal(n_layer_re(3, 0.5)[-1], obs_A)

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
        np.testing.assert_array_equal(np.round(n_layer_re(2, 1/3)[0],1), obs_temps)
        # here due to precision errors in rounding 1/3 use high tolerance all close
        np.testing.assert_allclose(n_layer_re(2, 1/3)[-1], obs_A, rtol=1E-7)

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
        np.testing.assert_array_equal(np.round(n_layer_re(1, 1)[0],1), obs_temps)
        np.testing.assert_array_equal(n_layer_re(1, 1)[-1], obs_A)

    def test_3layer_varepsilon(self):
        """
        Unit test to confirm that the variable epsilon
        method works as expected. This uses epsilon values
        of 0.9, 0.8 and 0.7 for the 3 atmospheric layers.
        """
        obs_temps = np.array([331.4,303.0,272.0,235.3])
        obs_A = np.array([[-1.   ,  1.   ,  0.1  ,  0.02 ],
                            [ 0.9  , -2.   ,  0.9  ,  0.18 ],
                            [ 0.08 ,  0.8  , -2.   ,  0.8  ],
                            [ 0.014,  0.14 ,  0.7  , -2.   ]])
        # note that we need to test the surface epsilon here because thats how I coded the function....
        np.testing.assert_equal(np.round(n_layer_re(3, [1.0,0.9,0.8,0.7])[0],1), obs_temps)
        np.testing.assert_allclose(n_layer_re(3, [1.0,0.9,0.8,0.7])[-1], obs_A, rtol=1E-7)
    
    def test_5layer_varepsilon(self):
        """
        Unit test to confirm the variable epsilon method works as 
        expected. This uses variable epsilon values of 0.1, 0.2, 
        0.3, 0.4 and 0.5. This does not test the A matrix, as I 
        didnt want to solve this by hand...
        """
        obs_temps = np.array([295.9,273.2,269.1,261.4,248.6,227.1])
        np.testing.assert_array_equal(np.round(n_layer_re(5, [1.0,0.1,0.2,0.3,0.4,0.5])[0],1), obs_temps)

### Question 3 - Lets add a parameterization for emissivity:  e <= 400mb = 0.1, 400 < p < 800 mb = 0.2, p > 800 e =0.3
def question_3(levs=LEVELS):
    """
    Function executes all the code necessary to complete question 3 in the lab manual.

    This plots the result of adding the parameterize emiss function to 
    determine the emissivity, and plots it against constant emissvity values of 
    each of the default parameters (e1 = 0.1, e2 = 0.2, e3 = 0.3)

    Also plots the dry and moist adiabatic lapse rates for reference.

    Parameters
    ----------
    levs : np.ndarray, default LEVELS
        levels over which to solve and plot data
    """
    # solve for each of the temperatures, can ignore the parameters here
    temps_emiss_param, _, _ = n_layer_re(levs)
    temps_e01, _, _ = n_layer_re(levs,epsilon=0.1)
    temps_e02, _, _ = n_layer_re(levs,epsilon=0.2)
    temps_e03, _, _ = n_layer_re(levs,epsilon=0.3)
    # calculate the lapse rates
    DALR = lapse_rate(levs, temps_emiss_param[0], 9.8)
    MALR = lapse_rate(levs, temps_emiss_param[0], 6.0)

    # plot the results - easier to just write plotting code here than
    # use a plotting function given how many lines we want to show
    fig, (ax,ax2) = plt.subplots(1,2,figsize=(14,10))
    ax.plot(temps_emiss_param,levs,c=IBM_PINK,lw=3,label=f"Parameterized {r'$\epsilon$'}")
    # plot the individual lines
    ax.plot(temps_e01,levs,c=IBM_PURPLE,lw=3,ls='--',label=f"{r'$\epsilon$'} = 0.1", alpha=0.7)
    ax.plot(temps_e02,levs,c=IBM_BLUE,lw=3,ls='--',label=f"{r'$\epsilon$'} = 0.2", alpha=0.7)
    ax.plot(temps_e03,levs,c=IBM_ORANGE,lw=3,ls='--',label=f"{r'$\epsilon$'} = 0.3", alpha=0.7)

    # plot the lapse rates
    ax2.plot(temps_emiss_param,levs,c=IBM_PINK,lw=3,label=f"Parameterized {r'$\epsilon$'}")
    ax2.plot(DALR,levs,c='burlywood',lw=3,ls=':',label='DALR')
    ax2.plot(MALR,levs,c='lightsteelblue',lw=3,ls=':',label='MALR')
    
    for axis in [ax,ax2]:
        format_profile(axis)
        axis.legend(frameon=False,prop={'weight':'bold'})

    ax.set_title('Temperature Profiles with Variable Emissivity', fontweight='bold', loc='left')
    ax2.set_title('Parameterized Emissivity Profile vs Lapse Rate', fontweight='bold', loc='left')
    fig.savefig(f'{OUTPATH}/q3_profiles.png')

### Question 4 - parameterize for ozone - key absorber in shortwave. compare the two profiles. why is this the case
def question_4(levs=LEVELS):
    """
    produces the figures necessary to answer question 4 in the lab manual.

    This creates 2 plots: one for comparing ozone with no ozone and another
    which compares the surface temperature when varying the tropopause height.

    Parameters
    ----------
    levs : np.ndarray
        levels on which to calculate the results
    """
    # part 1 - standard profile vs profile with ozone
    temps_std, _, _ = n_layer_re(levs) 
    temps_ozone, _, _ = n_layer_re(levs,ozone=True) 

    fig, ax = plt.subplots(1,1,figsize=(7,8))
    ax.plot(temps_std,levs,c=IBM_PINK,ls=':',lw=3,label=f"No Ozone")
    ax.plot(temps_ozone,levs,c=IBM_BLUE,lw=3,label=f"Ozone")
    ax.legend(frameon=False,prop={'weight':'bold'})
    format_profile(ax)
    ax.set_title('Impact of Ozone Parameterization on Profile',loc='left',fontweight='bold')
    fig.savefig(f'{OUTPATH}/q4_profiles.png')

    # now loop over tropopause heights and plot the surface temperature
    # as a function of tropopause height
    trop_heights = np.arange(400,50,-50)
    ozone_profiles = [n_layer_re(levs,ozone=True,strat=strat)[0] for strat in np.arange(400,50,-50)]
    # make nice colormap
    n_lines = len(trop_heights)
    cmap = mpl.colormaps['plasma_r']
    # take colors at regular intervals over colormap
    colors = cmap(np.linspace(0, 1, n_lines))

    # loop to plot
    fig1, ax1 = plt.subplots(1,1,figsize=(7,8))
    for th, color, profile in zip(trop_heights,colors,ozone_profiles):
        ax1.plot(profile,levs,c=color,lw=2,label=f'{th} hPa')
    format_profile(ax1)
    ax1.legend(frameon=False, prop={'weight':'bold'})
    ax1.set_title('Profile with Varying Tropopause Height',loc='left',fontweight='bold')
    fig1.savefig(f'{OUTPATH}/q4_trop.png')

### Question 5 - compare profiles for different latitudes
def question_5(levs=LEVELS):
    """
    Creates all the figures necessary to answer question 5 in the lab manual

    Plots the vertical atmospheric profile for different latitudes, accounting
    for the difference in tropopause height, albedo and insolation. This is kind of the
    most extreme example for each (e.g. polar ice cap, normal mid lats, equatorial ocean)

    Parameters
    ----------
    levs : np.ndarray, default LEVELS
        levels over which to calculate temperature
    """
    # set up a dictionary containing the attributes for each location
    attrs = {
            'North Polar Ice Cap' : {'S' : 190, 'albedo' : 0.6, 'strat' : 325},
            'Mid-Latitude Town'   : {'S' : 330, 'albedo' : 0.3, 'strat' : 250},
            'Equatorial Ocean'    : {'S' : 410, 'albedo' : 0.06, 'strat' : 120}
        }
    
    # loop to plot profiles
    fig1, ax1 = plt.subplots(1,1,figsize=(7,8))
    for (region, stats), color in zip(attrs.items(), [IBM_BLUE, IBM_PINK, IBM_ORANGE]):
        profile, _, _ = n_layer_re(levs,ozone=True,S=stats['S'],albedo=stats['albedo'],strat=stats['strat'])
        ax1.plot(profile,levs,c=color,lw=3,label=region)
    
    # do nice formatting
    ax1.set_title('Regional Variation in Annual Average Profile',fontweight='bold',loc='left')
    ax1.legend(frameon=False,prop={'weight':'bold'})
    format_profile(ax1,xmin=160)
    fig1.savefig(f'{OUTPATH}/q5_profiles.png')



def main():
    """
    Main function to run all experiments necessary to complete the
    lab assignment
    """
    question_3()
    question_4()
    question_5()

if __name__ == "__main__":
    print('testing n_layer_atmos solver')
    unittest.main(exit=False)
    # now execute the code for the class if everything works
    print('\nExecuting Lab Experiments')
    print('--------------------------')
    main()