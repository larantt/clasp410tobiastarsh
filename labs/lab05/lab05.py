#!/usr/bin/envs python3
"""
lab05.py - snowball earth
Author : Lara Tobias-Tarsh (laratt@umich.edu)

This lab uses the simple diffusion equation to build a basic
radiative transfer model used to explore how realistic a "snowball earth"
scenario is, and the stability of these snowball earth solutions. We
first construct our solver, then tune the parameters to reproduce a 
reference solution for Earth in its' current state. We then vary the 
the initial conditions and the solar multiplier explore how Earth's surface
temperature responds to these scenarios.

To execute the code in this lab:
```
    python lab04.py [--outpath PATH]
```
where PATH is an optional argument specifying where to save the output figures.
If no path is specified, figures will be saved in a new directory in your
current working directory.

The simulation should produce the following 5 figures:
    1. q1_validation.png - the figure from the lab manual for the verification step
    2. q2_curves.png - plots all the curves for varying the diffusivity and emissivity
    3. q2_optimization.png - plots a matrix showing optimization for the initial conditions
    4. q3_initial.png - three subplots for the initial conditions
    5. q4_gamma.png - plot to describe how temperature changes with gamma

in your terminal. Because a main() function is used, the code should
execute immediately and generate all the figures necessary to replicate
those shown in the lab report.
"""
from pathlib import Path
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# CONSTANTS
RADEARTH = 6378000.   # radius of the earth (m)
RHO = 1020.           # Density of sea-water (kg/m^3)
C = 4.2e6             # Heat capacity of water
SIGMA = 5.67e-8       # Stefan-Boltzmann constant
MXDLYR = 50.          # depth of mixed layer in (m)
EMISS = 1.            # emissivity of Earth (range 0-1)
ALBEDO_ICE = 0.6      # albedo of ice/snow
ALBEDO_WAT = 0.3      # albedo of water/ground
S0 = 1370.            # solar flux (W/m^2)   

# Define colors
IBM_BLUE = "#648FFF"
IBM_PURPLE = "#785EF0"
IBM_PINK = "#DC267F"
IBM_ORANGE = "#FE6100"
IBM_YELLOW = "#FFB000"

# set lambda (total diffusion coeffs)
LAMBDA_DEFAULT = 100.  # m^2/s, D/rho * C, D is heat diffusion coeff

def parse_args():
    """
    Parse command line arguments for the script.

    Function from Adi's github pull!
    
    Returns
    -------
    argparse.Namespace
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description='Solve and visualize permafrost melting using the heat equation.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--outpath',
        type=str,
        default=None,
        help='Directory path where figures should be saved. If not specified, creates a new directory in current working directory.'
    )
    return parser.parse_args()

def fix_latitude_labels(lats,ax):
    """
    Short helper function to fix the latitude labels
    for the zonal temperature profile plots

    Parameters
    -----------
    lats : np.ndarray
        array of latitudes from 0-180
    ax : plt.axes
        axis to fix the ticks on
    """
    up_tick = lats[:len(lats)//2] # get the first half of the ticks
    labels = np.concatenate((-1*up_tick[::-1],up_tick)) # combine and make negative
    ax.set_xticks(lats)
    ax.set_xticklabels(labels)
    # rotate the ticks for nice visualisation
    ax.tick_params(axis='x', labelrotation=45)

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

def mean_abs_err(model,obs):
    """
    Finds the mean absolute error, helper function
    to be used in the optimize_params function

    Parameters
    ----------
    model : np.ndarray
        results from the model to compare to obs
    obs : np.ndarray
        model reference solution
    """
    return np.mean(np.abs(obs - model))

def optimize_params(lam,eps,ref_sol,loss_fn=mean_abs_err,vis=False,debug=0):
    """
    Function optimizes the parameters for a model using a simple
    search methods to optimize the parameters used in the model relative
    to the reference solution provided.

    for where I got the unravel index things from, see examples in
    https://numpy.org/doc/stable/reference/generated/numpy.argmin.html where
    it does the ndarray thing.

    Parameters
    ----------
    lam : arrayLike
        parameter space for lamba
    eps : arrayLike
        parameter space for epsilon
    ref_sol : arrayLike
        the reference solution to optimize the model for
    loss_fn : callable, default mean_abs_err
        loss function to optimize 
    vis : bool, default FALSE
        should we visualize the combinations
    debug : int, default 0
        debugging and verbosity level

    Returns
    -------
    optimal_params : tuple->(lam, eps)
        optimal combination of lambda and epsilon for reference solution
    """
    # create grid of possible parameters for optimization
    lam_grid, eps_grid = np.meshgrid(lam,eps)
    # array to store results, fill with a fill value
    errors = []
    # now loop to compare the combinations
    for l, e in zip(lam_grid.flatten(),eps_grid.flatten()):
        # solve snowball earth
        _, model_temps = snowball_earth(lam=l,emiss=e)
        # get error for the reference solution
        error = loss_fn(model_temps,ref_sol)
        errors.append(error)
        if debug:
            print(f'error at lambda={l}, epsilon={e}: {error}')
    # now reshape to the coordinate grid for visualisation
    error_grid = np.reshape(np.array(errors),np.shape(lam_grid))
    # index in to get the best combination of indices
    optimal_idx = np.unravel_index(np.argmin(error_grid,axis=None),error_grid.shape)
    optimal_params = (lam_grid[optimal_idx],eps_grid[optimal_idx])
    print(f'Optimal Parameter Values:')
    print(f'-------------------------')
    print(f'Lambda  :    {optimal_params[0]}')
    print(f'Epsilon :    {optimal_params[1]}')
    print(f'Error   :    {error_grid[optimal_idx]}')
    # visualize if desired
    if vis:
        fig,ax = plt.subplots(1,1,figsize=(len(lam),len(eps)))
        # now visualize the data
        cax = ax.matshow(error_grid,cmap='plasma_r',alpha=0.5,vmax=100,vmin=0)

        cbar = fig.colorbar(cax, ax=ax, label='Mean Absolute Error',
                            shrink=0.8,pad=0.05,extend='max')

        ax.set_xlabel(r'Diffusivity - $\lambda$ (m$^{2}$/s)',fontsize=12,fontweight='bold')
        ax.set_ylabel(r'Emissivity - $\epsilon$',fontsize=12,fontweight='bold')

        ax.set_title(r'Mean Absolute Error Optimization for $\lambda$ and $\epsilon$',
                     fontsize=14,fontweight='bold',loc='left')
        
        ax.set_title(f'Optimum Parameters: $\\lambda$={optimal_params[0]:.1f}, $\\epsilon$={optimal_params[1]:.1f}',
                     fontsize=12,fontweight='bold',loc='right',color='dimgrey')

        # set the ticks to represent each value, round to 1dp
        ax.set_xticks(np.arange(len(lam)))
        ax.set_xticklabels(np.round(lam,1))
        ax.set_yticks(np.arange(len(eps)))
        ax.set_yticklabels(np.round(eps,1))
        ax.tick_params(axis='x', bottom=True, top=False, labelbottom=True, labeltop=False)

        # annotate each box
        for (i, j), err in np.ndenumerate(error_grid):
            ax.text(j, i, f"{err:.2f}", va="center", ha="center",fontsize=10)

        bold_axes(ax)
        fig.tight_layout()
        fig.savefig(f'{OUTPATH}/q2_optimization.png')

    return optimal_params

def gen_grid(nbins=18):
    """
    Generate a grid from 0 to 180 lat (where 0 is the south pole, 180 is the north)
    where each returned point represents the cell center.

    Parameters
    ----------
    nbins : int, defaults to 18
        set the number of latitude bins
    
    Returns
    -------
    dlat : float
        Grid spacing in degrees
    lats : np.array
        Array of cell center latitudes
    """
    dlat = 180 / nbins   # latitude spacing
    lats = np.linspace(dlat/2., 180-dlat/2, nbins) # create the grid

    return dlat, lats

def temp_warm(lats_in):
    '''
    Create a temperature profile for modern day "warm" earth.

    Parameters
    ----------
    lats_in : np.ndarray
        Array of latitudes in degrees where temperature is required.
        0 corresponds to the south pole, 180 to the north.
    
    Returns
    -------
    temp : np.ndarray
        Temperature in Celcius.
    '''
    # Set initial temperature curve
    T_warm = np.array([-47, -19, -11, 1, 9, 14, 19, 23, 25, 25,
    23, 19, 14, 9, 1, -11, -19, -47])
    # Get base grid:
    npoints = T_warm.size
    dlat = 180 / npoints # Latitude spacing.
    lats = np.linspace(dlat/2., 180-dlat/2., npoints) # Lat cell centers.
    # Fit a parabola to the above values
    coeffs = np.polyfit(lats, T_warm, 2)
    # Now, return fitting sampled at "lats".
    temp = coeffs[2] + coeffs[1]*lats_in + coeffs[0] * lats_in**2

    return temp

def insolation(S0, lats):
    '''
    Given a solar constant (`S0`), calculate average annual, longitude-averaged
    insolation values as a function of latitude.
    Insolation is returned at position `lats` in units of W/m^2.

    Parameters
    ----------
    S0 : float
        Solar constant (1370 for typical Earth conditions.)
    lats : Numpy array
        Latitudes to output insolation. Following the grid standards set in
        the diffusion program, polar angle is defined from the south pole.
        In other words, 0 is the south pole, 180 the north.

    Returns
    -------
    insolation : numpy array
        Insolation returned over the input latitudes.
    '''

    # Constants:
    max_tilt = 23.5   # tilt of earth in degrees

    # Create an array to hold insolation:
    insolation = np.zeros(lats.size)

    #  Daily rotation of earth reduces solar constant by distributing the sun
    #  energy all along a zonal band
    dlong = 0.01  # Use 1/100 of a degree in summing over latitudes
    angle = np.cos(np.pi/180. * np.arange(0, 360, dlong))
    angle[angle < 0] = 0
    total_solar = S0 * angle.sum()
    S0_avg = total_solar / (360/dlong)

    # Accumulate normalized insolation through a year.
    # Start with the spin axis tilt for every day in 1 year:
    tilt = [max_tilt * np.cos(2.0*np.pi*day/365) for day in range(365)]

    # Apply to each latitude zone:
    for i, lat in enumerate(lats):
        # Get solar zenith; do not let it go past 180. Convert to latitude.
        zen = lat - 90. + tilt
        zen[zen > 90] = 90
        # Use zenith angle to calculate insolation as function of latitude.
        insolation[i] = S0_avg * np.sum(np.cos(np.pi/180. * zen)) / 365.

    # Average over entire year; multiply by S0 amplitude:
    insolation = S0_avg * insolation / 365

    return insolation

def snowball_earth(nbins=18, dt=1, tstop=10000, debug=0, lam=100., temp=temp_warm, dynamic_albedo=False,
                   spherecorr = True, initialise_only=False, albedo=0.3, emiss=1., s0=1370.,gamma=1.0):
    """
    Function is a solver for snowball earth. This is the main code for the lab. 
    It initialises a simulation with a set number of bins and either creates
    of uses supplied initial conditions. It then calculates the physical and radiative 
    parameters and runs the simulation. It can correct for spherical coordinates, 
    radiative terms, and use a dynamic albedo.

    Inputs
    ------
    nbins : int, defaults to 18
        Number of latitude bins
    dt : float, defaults to 1
        Timestep in units of years
    tstop : float, defaults to 10,000
        stop time in years
    debug : int, default 0 (False)
        set verbosity of the debug statements
    lam : float, defaults to 100
        diffusion coefficient of ocean in m^2/s (D/rho * C)
    temp : any, default temp_warm
        either a function, an array, a float or an int to initialise solver
    dynamic_albedo : bool
        should dynamic albedo across lats be used?
    spherecorr : bool, True
        use spherical coordinates
    initialise_only : bool, defaults to False
        returns the initial conditions only and does not solve in time
    albedo : float, defaults to 0.3
        set the Earth's albedo
    emiss : float, defaults to 1
        set ground emissivity
    s0 : float, defaults to 1370
        set incoming solar forcing constant
    gamma : float, defaults to 1
        the solar multiplier for insolation

    Returns
    -------
    lats : np.array
        Latitude grid in degrees where 0 is the south pole
    temp : np.array
        array of temperatures
    """
    # timestep from years to seconds 
    dt_sec = 365 * 24 * 3600 * dt

    if debug > 2:
        print("Generating grid...")
    # call gen_grid
    dlat, lats = gen_grid()
    # latitude correction
    dy = (dlat/180) * np.pi * RADEARTH


    # type checking! - is it a fixed type? if so, make array
    if isinstance(temp,(int,float)):
        temp = np.full_like(lats,temp,dtype=np.float64)
    elif isinstance(temp, np.ndarray):
        # check it has the same size as lats
        if np.shape(temp) != np.shape(lats):
            raise(ValueError(f'temp with shape {np.shape(temp)} must match {np.shape(lats)}'))
    else:
        # try calling as a function
        try:
            temp = temp(lats)
        except:
            raise(TypeError(f'type {type(temp)} is incorrect. Pass temp = float, int, callable(np.array) or np.ndarray of shape {np.shape(lats)}'))

    if initialise_only:
        return lats, temp

    # number of timesteps
    nstep = int(tstop/dt)
    # debug for initialization
    if debug > 1:
        print("Debugging initialisation...")
        print(f"Function called for nbins={nbins}, dt={dt}, tstop={tstop}")
        print(f"Results in nstep={nstep} timesteps, dy={dy}")
        print("resulting lat grid:")
        print("--------------------------------------------------------")
        print(lats)

    if debug > 2:
        print(f"Building matrices...")  

    # create A matrix (square matrix nbins * nbins)
    A_matrix = np.zeros((nbins,nbins))
    # set the points on diagonals to 2
    A_matrix[np.arange(nbins),np.arange(nbins)] = -2
    # set off diagonal elements
    A_matrix[np.arange(nbins-1),np.arange(nbins-1)+1] = 1
    A_matrix[np.arange(nbins-1)+1,np.arange(nbins-1)] = 1
    # set boundary conditions (presumably could change later)
    A_matrix[0,1], A_matrix[-1,-2] = 2, 2
    A_matrix /= dy**2

    # create the L matrix
    L_matrix = np.identity(nbins) - (dt_sec * lam * A_matrix)
    L_inv = np.linalg.inv(L_matrix)

    # create the B matrix
    B_matrix = np.zeros((nbins,nbins))
    # set off diagonal elements
    B_matrix[np.arange(nbins-1),np.arange(nbins-1)+1] = 1
    B_matrix[np.arange(nbins-1)+1,np.arange(nbins-1)] = -1
    B_matrix[0,:], B_matrix[-1,:] = 0, 0

    # calculate Axz
    A_xz = np.pi * ((RADEARTH+50.0)**2 -RADEARTH**2) * np.sin(np.pi/180.*lats)
    dA_xz = np.matmul(B_matrix,A_xz) / (A_xz * 4 * dy**2)

    if debug > 1:
        print("A Matrix:")
        print(A_matrix)
        print("L Matrix:")
        print(L_matrix)
        
    if debug > 2:
        print("Time integrating...")

    # go in time
    ins = gamma * insolation(s0,lats)
    for i in range(nstep):
        # add spherical correction term
        if spherecorr:
            temp += dt_sec * lam * dA_xz * np.matmul(B_matrix, temp)

        # albedo calculations
        if dynamic_albedo:
            albedo = np.zeros_like(temp)
            loc_ice = temp <= -10
            albedo[loc_ice] = ALBEDO_ICE
            albedo[~loc_ice] = ALBEDO_WAT
        
        # calculate the radiative forcing
        radiative = (1-albedo) * ins  - emiss*SIGMA*(temp+273.15)**4
        # update temperatures
        temp += dt_sec * radiative / (RHO*C*MXDLYR)
        # matrix multiplication for final temperature in the loop
        temp = np.matmul(L_inv,temp)

    return lats, temp

def snowball_test():
    """
    Reproduces the test figure in Lab 5.

    Using the default values for gridsize, diffusion etc. and a 
    warm earth initial condition, plot:
        - initial condition
        - simple diffusion only
        - simple diffusion + spherical correction
        - simple diffusion + spherical correction + insolation

    Then saves figure to the directory specified in the run command.
    """
    # Generate very simple grid
    lats, initial = snowball_earth(initialise_only=True)
    lats2, t_diff = snowball_earth(spherecorr=False,s0=0, emiss=0)
    lats3, t_sphere = snowball_earth(s0=0, emiss=0)
    lats4, t_sphe = snowball_earth()

    # make plot
    fig,ax = plt.subplots(1,1,figsize=(12,8))
    ax.plot(lats,initial,label='Warm Earth Initial Conditions',lw=3, c=IBM_BLUE)
    ax.plot(lats2,t_diff, label='Simple heat diffusion',lw=3, c=IBM_PINK)
    ax.plot(lats3,t_sphere, label='Simple heat diffusion + Spherical Correction',lw=3, c=IBM_PURPLE)
    ax.plot(lats4,t_sphe, label='Simple heat diffusion + Spherical Correction + Radiative',lw=3, c=IBM_YELLOW)

    # format plot
    ax.set_xlabel('Latitude ($^{\\circ}$)',fontweight='bold')
    ax.set_ylabel('Temperature  ($^{\\circ} C$)',fontweight='bold')

    # set the ticks to make conventional sense
    fix_latitude_labels(lats,ax)

    # finish formatting
    ax.legend(frameon=False,ncols=2,prop={'weight': 'bold','size':'8'})
    bold_axes(ax)
    ax.set_title('Snowball Earth Solver Validation',loc='left',fontweight='bold',fontsize=14)
    fig.savefig(f'{OUTPATH}/q1_validation.png')

def plot_variation(ref_sol, results, labels, lats, ax):
    """
    Function plots curves on line plot to compare to a 
    reference solution by iterating over a set of colors and solutions.

    Parameters
    ----------
    ref_sol : np.ndarray
        array for the reference solution
    results : list(np.ndarray)
        list of arrays with the results to plot for comparison
    labels : arrayLike
        labels for each solution in results
    lats : np.ndarray
        latitudes for the x axis of the plot
    ax : mpl.axes
        axes object to plot results on

    """
    # set up a colormap
    n_lines = len(results)
    cmap = mpl.colormaps['plasma']
    # take colors at regular intervals over colormap
    colors = cmap(np.linspace(0, 1, n_lines))
    # if the labels are floats, string format
    try:
        [ax.plot(lats,sol,lw=3,label=f'{lab:.2f}',c=col) for sol,lab,col in zip(results,labels,colors)]
    except:
        # if the labels arent string formattable, dont.
        [ax.plot(lats,sol,lw=3,label=lab,c=col) for sol,lab,col in zip(results,labels,colors)]

    # format figure
    ax.plot(lats,ref_sol,c='darkgrey',lw=3,ls='--',label='Reference')
    ax.legend(ncols=3,loc='upper right',prop={'weight': 'bold','size':'8'},frameon=False)
    ax.set_ylabel('Temperature  ($^{\\circ} C$)',fontweight='bold')
    
def question_two():
    """
    Tunes the model to produce warm earth equilibrium by 
    varying λ from 0 to 150 and ε from 0 to 1.

    This function will first explore the impact on the equilibrium 
    solution independently, then will select the combination that best
    reproduces the solution. This is done using a grid search algorithm
    which optimizes the mean absolute error as the loss function.

    Returns nothing, but saves a heatmap from optimization and a
    set of line plots showing the temperatures from varying each
    parameter.
    """
    # first create the temp_warm curve for warm earth equilibrium
    lats, _ = snowball_earth(initialise_only=True)
    warm_earth_equilib = temp_warm(lats)

    # first vary lambda
    curves_lam, dts_lam = [], []
    lambdas = np.arange(0,175,25)
    # now loop and append curves to list
    for val in lambdas:
        _, curve = snowball_earth(lam=val)
        del_t = warm_earth_equilib - curve
        curves_lam.append(curve)
        dts_lam.append(del_t)
    
    # next vary epsilon
    curves_eps, dts_eps = [], []
    epsilons = np.arange(0.0,1.1,0.2)
    # now loop and append curves to list
    for val in epsilons:
        _, curve = snowball_earth(emiss=val)
        del_t = warm_earth_equilib - curve
        curves_eps.append(curve)
        dts_eps.append(del_t)
    
    # now optimize the parameter space for both
    (optimum_l, optimum_e) = optimize_params(np.arange(0,160,10),np.arange(0,1.1,0.1),warm_earth_equilib,vis=True)
    # finally plot full solution for comparison
    _, optimal_temp = snowball_earth(emiss=optimum_e,lam=optimum_l)

    fig,ax = plt.subplots(3,1,figsize=(15,12),sharex=True)
    plot_variation(warm_earth_equilib,curves_lam,lambdas,lats,ax[0])
    ax[0].set_title('Varying Lambda',fontsize=12,fontweight='bold',loc='left')

    plot_variation(warm_earth_equilib,curves_eps,epsilons,lats,ax[1])
    ax[1].set_title('Varying Epsilon',fontsize=12,fontweight='bold',loc='left')

    ax[2].plot(lats,optimal_temp,lw=3,c=IBM_PINK,label='Optimum Model Solution')
    ax[2].plot(lats,optimal_temp,lw=3,ls='--',c='darkgrey',label='Reference')
    ax[2].legend(ncols=1,frameon=False,loc='upper right',prop={'weight': 'bold','size':'8'})
    ax[2].set_title(f'Optimum Parameters: $\\lambda$={optimum_l:.1f}, $\\epsilon$={optimum_e:.1f}',
                    fontweight='bold',fontsize=10,loc='left')
    ax[2].set_xlabel('Latitude ($^{\\circ}$)',fontweight='bold')
    ax[2].set_ylabel('Temperature ($^{\\circ}$C)',fontweight='bold')
    
    [bold_axes(ax) for ax in ax.flatten()]
    [fix_latitude_labels(lats,ax) for ax in ax.flatten()]

    fig.tight_layout()
    fig.savefig(f"{OUTPATH}/q2_curves.png")
    
def question_three(epsilon_q3=0.7,lambda_q3=40.):
    """
    Performs Question 3 in the lab, which explores three scenarios
        1. cold earth, where temperatures in all locations are -60C
        2. hot earth, where temperatures in all locations are 60C
        3. flash freeze, where there is an albedo of 0.6, causing a rapid freeze

    This function saves a figure with each of the solutions for this problem

    Parameters
    ----------
    epsilon_q3 : float, default 0.7
        value of epsilon to use in this function
    lambda_q3 : float, default 40.
        value of lambda to use in this function

    Note that these parameters were derived from running question two for 
    optimization.
    """
    # hot earth (60◦ at all locations)
    lats, hot_earth_init = snowball_earth(lam=lambda_q3,emiss=epsilon_q3,temp=60.,dynamic_albedo=True,initialise_only=True)
    _, hot_earth_10 = snowball_earth(lam=lambda_q3,emiss=epsilon_q3,temp=60.,dynamic_albedo=True)
    _, hot_earth_20 = snowball_earth(lam=lambda_q3,emiss=epsilon_q3,temp=60.,dynamic_albedo=True,tstop=20000)
    # cold earth (-60◦ at all locations)
    _, cold_earth_init = snowball_earth(lam=lambda_q3,emiss=epsilon_q3,temp=-60.,dynamic_albedo=True,initialise_only=True)
    _, cold_earth_10 = snowball_earth(lam=lambda_q3,emiss=epsilon_q3,temp=-60.,dynamic_albedo=True)
    _, cold_earth_20 = snowball_earth(lam=lambda_q3,emiss=epsilon_q3,temp=-60.,dynamic_albedo=True,tstop=20000)
    # flash freeze - albedo of 0.6
    _, flash_earth_init = snowball_earth(lam=lambda_q3,emiss=epsilon_q3,albedo=0.6,dynamic_albedo=False,initialise_only=True)
    _, flash_earth_10 = snowball_earth(lam=lambda_q3,emiss=epsilon_q3,albedo=0.6,dynamic_albedo=False)
    _, flash_earth_20 = snowball_earth(lam=lambda_q3,emiss=epsilon_q3,albedo=0.6,dynamic_albedo=False,tstop=20000)

    # plotting
    fig,ax = plt.subplots(1,3,figsize=(18,5),sharey=True)

    ax[0].plot(lats,hot_earth_init,c=IBM_ORANGE,lw=3,alpha=0.5,ls='--',label='initial conditions')
    ax[0].plot(lats,hot_earth_10,c=IBM_ORANGE,lw=3,alpha=0.5,label='10,000 years')
    ax[0].plot(lats,hot_earth_20,c=IBM_ORANGE,lw=3,label='20,000 years')
    ax[0].axhline(-10,c='k',lw=3,label='-10$^{\\circ}$C',ls='--')

    ax[1].plot(lats,flash_earth_init,c=IBM_PURPLE,lw=3,alpha=0.5,ls='--',label='initial conditions')
    ax[1].plot(lats,flash_earth_10,c=IBM_PURPLE,lw=3,alpha=0.5,label='10,000 years')
    ax[1].plot(lats,flash_earth_20,c=IBM_PURPLE,lw=3,label='20,000 years')
    ax[1].axhline(-10,c='k',lw=3,label='-10$^{\\circ}$C',ls='--')

    ax[2].plot(lats,cold_earth_init,c=IBM_BLUE,lw=3,alpha=0.5,ls='--',label='initial conditions')
    ax[2].plot(lats,cold_earth_10,c=IBM_BLUE,lw=3,alpha=0.5,label='10,000 years')
    ax[2].plot(lats,cold_earth_20,c=IBM_BLUE,lw=3,label='20,00 years')
    ax[2].axhline(-10,c='k',lw=3,label='-10$^{\\circ}$C',ls='--')

    for axs,title in zip(ax.flatten(),['Hot Earth','Flash Freeze','Cold Earth']):
        axs.set_title(title,loc='left',fontweight='bold',fontsize=12)
        axs.legend(frameon=False,prop={'weight': 'bold','size':'8'},ncols=2)
        axs.set_xlabel('Latitude ($^{\\circ}$)',fontweight='bold')
        fix_latitude_labels(lats,axs)
        bold_axes(axs)
    
    ax[0].set_ylabel('Temperature ($^{\\circ}$C)',fontweight='bold')
    
    fig.suptitle(f'Impact of Initial Conditions on Zonal Surface Temperature',fontsize=16,fontweight='bold')
    fig.tight_layout()
    fig.savefig(f'{OUTPATH}/q3_initial.png')

def question_four(epsilon_q4=0.7,lambda_q4=40.):
    """
    Question four adds a solar multiplier to the solver, then plots the model as a 
    function of gamma in order to determine stability.

    Parameters
    ----------
    epsilon_q4 : float, default 0.7
        value of epsilon to use in this function
    lambda_q4 : float, default 40.
        value of lambda to use in this function

    Note that these parameters were derived from running question two for 
    optimization.
    """
    # initialize the simulation for cold earth
    lats, temps = snowball_earth(lam=lambda_q4,emiss=epsilon_q4,
                                      temp=-60.,dynamic_albedo=True,gamma=0.4)
    # set up loop
    gammas = np.arange(0.4,1.405,0.05)
    gammas = np.concatenate((gammas[0:-1],gammas[::-1]),axis=None) # concat to go up and down
    # loop up
    mean_temps = []
    for gam in gammas:
        _, t = snowball_earth(lam=lambda_q4,emiss=epsilon_q4,temp=temps,
                              dynamic_albedo=True,gamma=gam)
        # update temps
        temps = t
        mean_temps.append(np.mean(t))
    
    fig,ax = plt.subplots(1,1,figsize=(12,8))
    ax.plot(gammas,mean_temps,c=IBM_BLUE,lw=3)
    ax.set_xlabel('Gamma',fontweight='bold',fontsize=12)
    ax.set_ylabel('Temperature ($^{\\circ}$C)',fontweight='bold',fontsize=12)
    ax.set_title('Mean Global Surface Temperature as a Function of Solar Multiplier',
                 loc='left',fontweight='bold',fontsize=14)
    bold_axes(ax)
    fig.savefig(f'{OUTPATH}/q4_gamma.png')

def main():
    """
    Main function to execute the code in this lab
    """
    # create path for figures to be saved if not already existing
    args = parse_args()
    global OUTPATH
    if args.outpath:
        OUTPATH = args.outpath
    else:
        # will create a figures directory in your current working directory
        OUTPATH = f'{os.getcwd()}/laratt_lab05_figures'
    
    # make the output directory for the figures in the lab
    Path(OUTPATH).mkdir(parents=True, exist_ok=True)

    # now run the lab experiments
    print('Running Validation (Question 1)')
    snowball_test()
    print('Running parameter tuning (Question 2)')
    question_two()
    print('Varying initial conditions (Question 3)')
    question_three()
    print('Varying solar multiplier (Question 4)')
    question_four()
    print(f'Complete! Figures saved to {OUTPATH}')

if __name__ == "__main__":
    main()