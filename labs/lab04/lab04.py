#usr/bin/envs python3
"""
Lara Tobias-Tarsh (laratt)
lab04.py - permafrost melting

This lab explores how forward differencing methdos can be used
to solve the heat equation with time in 1 dimension. We use this
heat diffusion equation model to answer questions about the seasonal
dynamics of permafrost and the impact of warming on permafrost in
the future.

To execute the code in this lab simply run:
```
    python3 lab04.py
```

in your terminal. Because a main() function is used, the code should
execute immediately and generate all the figures necessary to replicate
those shown in the lab report. If a directory is not supplied in the
OUTPATH global variable as user input, the figures in the lab will
save in a new directory created in your current working directory.
"""
import os
import logging
import warnings
from pathlib import Path
import unittest
import numpy as np
import matplotlib.pyplot as plt
import datetime

# Kangerlussuaq average temperature:
T_KANGER = np.array([-19.7, -21.0, -17., -8.4, 2.3, 8.4,
10.7, 8.5, 3.1, -6.0, -12.0, -16.9])

# define nice gradient blue red color pallete (red, blue)
COLORS = [("#d5b3b3","#96b6f2"),
          ("#bc8484","#5689ea"),
          ("#a15858","#1b5ede"),
          ("#733f3f","#13439f"),
          ("#452525","#0b285f")]

# define nice colorblind friendly plotting colors
# source: https://davidmathlogic.com/colorblind/#%23648FFF-%23785EF0-%23DC267F-%23FE6100-%23FFB000
IBM_CMAP = ["#648FFF","#785EF0","#DC267F","#FE6100","#FFB000"]

OUTPATH = "/Users/laratobias-tarsh/Documents/fa24/clasp410tobiastarsh/labs/lab04/figures"

# initial configuration for debugging (changed dynamically in heatdiff)
logging.basicConfig(level=logging.WARNING)

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

def default_init(xgrid):
    """
    Function to default initialise the U array using
    4*x - 4*(x**2).

    Note that this doesn't have to be the case, and the
    user could define any function they wanted to pass in
    here.

    Parameters
    -----------
    xgrid : np.array
        1D array representing the grid spacing in the x direction

    """
    return 4*xgrid - 4*xgrid**2

def default_bounds(u_arr,**kwargs):
    """
    Default function for enforcing boundary conditions
    in a simulation.

    In this case, we set the upper and lower boundary
    to 0 in both directions.

    Parameters : u_arr
        U array used in the heat equation
    """
    # set the top boundary conditions to 0
    u_arr[0, :] = 0
    # set the bottom boundary conditions to 0
    u_arr[-1, :] = 0

def kanger_bounds(u_arr,time_arr,shift=0.,temp=T_KANGER):
    """
    Helper function to set the boundary
    conditions for Kangerlussuaq, Greenland.

    In this case, the upper boundary conditions are set
    using the timeseries of daily temperatures generated by
    the temp_kanger function, interpolated to whatever the
    unit of time in the solver is.

    The lower boundary conditions are set to be constant at
    5C, which represents geothermal warming.

    Parameters
    ----------
    u_arr : np.ndarray
        2D U array in the heat equation solver
    time_arr : np.array
        array of times in days
    """
    def temp_kanger(t,temp=temp,shift=shift):
        '''
        For an array of times in days, return timeseries of temperature for
        Kangerlussuaq, Greenland.
        '''
        t_amp = (temp - temp.mean()).max()
        return (t_amp*np.sin(np.pi/180 * t - np.pi/2) + temp.mean()) + shift
    
    # time is in seconds, 1 day = 86400 seconds so convert time arr to days
    # get daily timeseries and interpolate to dt resolution
    temps = temp_kanger(time_arr/86400)
    # now set upper bounds
    u_arr[0, :] = temps
    # set lower bounds to 5
    u_arr[-1, :] = 5

def apply_bounds(func, **kwargs):
    """
    Helper function that applies boundary conditions
    with an uncertain number of arguments where needed.
    """
    func(**kwargs)

def heatmap(u_arr,t_arr,d_arr,units=None,ax=None,fig=None,title='Heatmap',diverge=True):
    """
    Function makes heatmap plot to display the variation of
    heat with depth and time.

    Parameters
    ----------
    u_arr : np.ndarray
        heat array with time
    t_arr : np.array
        array of times for the simulation
    d_arr : np.array
        array of depths for the simulation
    units : str, defaults to seconds
        Units to convert the time into, assumes seconds as a base unit
    ax : mpl.Axes, defaults to None
        which axes to plot on if supplied, otherwise creates axes
    fig : mpl.Figure, defaults to None
        which figure to plot on if supplied, otherwise creates figure
    title : str, defaults to 'Heatmap'
        title for the axis
    diverge : bool, defaults to True
        whether to set a diverging colormap over the maximum range of values
        centered at 0 or not.

    Returns
    -------
    fig : mpl.Figure
        figure with the plot
    ax : mpl.Axes
        axes with the plot

    """
    # create figure and axes if they arent already passed
    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(12,8))

    # convert times to desired units
    if units == 'days':
        t_arr_scaled = t_arr/86400
    elif units == 'years':
        t_arr_scaled = (t_arr/86400)/365
    elif units == 'seconds':
        t_arr_scaled = t_arr
    else:
        warnings.warn(f"No or invalid time units: {units} specified.\nChoose days, years, or seconds",RuntimeWarning)
        t_arr_scaled = t_arr

    # determine the largest absolute value of the min and max to the nearest 5
    scale_range = 5 * ((np.max([np.abs(u_arr.min()),np.abs(u_arr.max())]) + 4) // 5)
    # make p-color plot
    if diverge:
        cmap = ax.pcolor(t_arr_scaled,d_arr,u_arr,cmap='seismic',vmin=-1*scale_range,vmax=scale_range)
    else:
        cmap = ax.pcolor(t_arr_scaled,d_arr,u_arr,cmap='cividis_r')
    # set colorbar
    cb = fig.colorbar(cmap, ax=ax, label='Temperature ($C$)')
    # flip axes to decrease depth with height
    ax.invert_yaxis()
    ax.set_ylabel('depth (m)',fontweight='bold',fontsize=12)
    ax.set_xlabel(f'time ({units})',fontweight='bold',fontsize=12)
    bold_axes(ax)
    ax.set_title(title,fontweight='bold',fontsize=16,loc='left')

    return fig, ax

def heatdiff(xmax, tmax, dx, dt, bconds=default_bounds, init=default_init, c2=1, debug=0):
    '''
    Function to solve the 1D heat diffusion equation in time.

    This function first checks the stability of the input parameters (if they are not stable, errors).
    It then initialises the simulation using user defined or preset functions for initial and
    boundary conditions, and loops in time for a minimum of 2 years (or until the simulation is complete).

    After 2 years, a covergence check sequence is initiated at the end of each year to see if the
    temperature in the isothermal zone has been constant. If so, the simulation terminates.

    Parameters:
    -----------
    xmax : float
        maximum depth of the ground in the x direction
    tmax : float
        maximum amount of time in seconds
    dx : float
        grid spacing in the x direction
    dt : float
        time step in seconds
    bconds : function
        user defined function to enforce boundary conditions
        in the simulation.
        Defaults to default_bounds(), sets to 0
    init : function
        user defined function to set initial conditions on
        the heat grid.
        Defaults to default_init(), sets to 4x - 4x^2
    c2 : float
        coefficient of heat diffusion
    debug : int
        verbosity of debugging output.
        Defaults to 0/False (No output)

    Returns:
    --------
    xgrid : np.ndarray
        array representing depth
    tgrid : np.ndarray
        array representing time
    U : np.ndarray
        array representing temperature in the simulation

    '''
    # set logging levels to print debug output
    if debug > 0:
        logging.getLogger().setLevel(logging.INFO)
        logging.info('Debugging turned on (low verbosity)')
    elif debug > 1:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.info('Debugging turned on (high verbosity)')

    # perform initial check for stability (first order in time but 2nd order in space)
    if dt > (dx**2)/(2 * c2):
        raise(ValueError(f'Initial conditions dx : {dx}, dt : {dt}, c2 :{c2} are unstable. Reduce dt for stability.'))
    
    # Start by calculating size of array: MxN
    M = int(np.round(xmax / dx + 1))
    N = int(np.round(tmax / dt + 1))

    xgrid, tgrid = np.arange(0, xmax+dx, dx), np.arange(0, tmax+dt, dt)

    # debug statements
    logging.info('Our grid goes from 0 to %f m and 0 to %f s' % (xmax,tmax))
    logging.info('Our spatial step is %f and time step is %f' % (dx,dt))
    logging.info('There are %f points in space and %f points in time.' % (M,N))
    logging.info('Here is our spatial grid:')
    logging.info(xgrid)
    logging.info('Here is our time grid:')
    logging.info(tgrid)

    # Initialize our data array:
    U = np.zeros((M, N))

    # if function to initialize passed in call
    if callable(init):
        U[:, 0] = init(xgrid)
    else:
        # type check for int or float
        if not isinstance(init,(int,float)):
            raise TypeError(f'value for init {init}, type {type(init)} must be callable, int or float')
        # otherwise, set to init
        U[:, 0] = init
    
    # enforce boundary conditions - note this MUST be a function because in theory
    # there could be conditions at any four boundaries dependent on any number of 
    # internal cells/processes/dynamics
    if not callable(bconds):
        raise TypeError(f'Value for bconds: {bconds}, type: {bconds} must be callable')
    
    apply_bounds(bconds,u_arr=U,time_arr=tgrid)
    #bconds(U,time_arr=tgrid)

    # Set our "r" constant.
    r = c2 * dt / dx**2

    # Solve! Forward differnce ahoy.
    # loop in time
    for j in range(N-1):
        U[1:-1, j+1] = (1-2*r) * U[1:-1, j] + r*(U[2:, j] + U[:-2, j])
        # add additional debug statement if asked (using logger avoids slow conditionals)
        logging.debug('Set U[1:-1, j+1] to:')
        logging.debug(U[1:-1, j+1])
        # re-enforce boundary conditions - note that for this lab this does nothing
        # but in theory if we had Dirichlet or Neumann boundary conditions we would
        # need to potentially reinforce them
        #bconds(U,time_arr=tgrid)
        apply_bounds(bconds,u_arr=U,time_arr=tgrid)
        # space to add convergence check 
        # (break loop if the average gradient of the isothermal region hardly changes over yr)

    # Return grid and result:
    return xgrid, tgrid, U

def plot_all_profiles(us,ds,dt,ax,shift,colors,title='Profile of Permafrost'):
    """
    Make nice gradient profile for multiple temperatures
    """
    # 1 year in seconds
    y2s = 365*5*86400
    # now set indexing to get final year
    loc = int(-y2s/dt)

    # now loop and plot
    for u_arr, d_arr,cs,s in zip(us,ds,colors,shift):
        # get summer and winter values
        winter = u_arr[:, loc:].min(axis=1)
        summer = u_arr[:, loc:].max(axis=1)
        # do plotting
        ax.plot(summer,d_arr,c=cs[0],label=f'Summer ({s}$^\circ$C warming)',lw=3)
        ax.plot(winter,d_arr,c=cs[1],label=f'Winter ({s}$^\circ$C warming)',lw=3)
        ax.set_ylim(0,60)  # change later to actually find the bottom of the coldest profile
        ax.invert_yaxis()
        ax.set_xlabel(r'Temperature ($^\circ$C)',fontsize=12,fontweight='bold')
        ax.set_ylabel('Depth (m)',fontsize=12,fontweight='bold')
        ax.legend(loc='lower left',frameon=False,ncols=2)

    ax.axvline(0,ls='--',c='lightgrey',label=r'0$^\circ$C')
    ax.set_title(title,loc='left',fontsize=14,fontweight='bold')
    bold_axes(ax)

def plot_depth_thickness(us,ds,dt,ax,shift,colors,title='Thickness and Depth of Permafrost'):
    """
    Function makes a bar plot of depth and thickness of permafrost
    """
    # 1 year in seconds
    y2s = 365*5*86400
    # now set indexing to get final year
    loc = int(-y2s/dt)

    # make an empty dictionary to store output
    perm_data = {}
    # now loop and plot
    for u_arr, d_arr,s in zip(us,ds,shift):
        # get summer and winter values
        summer = u_arr[:, loc:].max(axis=1)
        # determine the depth (maximum depth that isothermal zone = 0)
        # last instance of 0 in the curve before geothermal heating
        depth_idx = np.argwhere(summer <= 0.)[-1][0]   # note, second index ensures the depth index is retrived
        depth = d_arr[depth_idx]
        # determine the thickness (depth of active layer - depth)
        # first index below 0 in summer will define active layer
        active_idx = np.argwhere(summer <= 0)[0][0] # note, second index ensures the depth index is retrived
        thickness = d_arr[depth_idx] - d_arr[active_idx]
       
        perm_data[f'+{s} warming'] = (depth,thickness)

    # now plotting
    labels = ['depth (m)','thickness (m)']
    x = np.arange(len(labels))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    for lab,measurement in perm_data.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=lab)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    #ax.set_xticks(i+width,['Depth','Thickness']) 
    ax.legend(loc='lower left',frameon=False,ncols=2)

    ax.set_title(title,loc='left',fontsize=14,fontweight='bold')
    bold_axes(ax)



def permafrost_solve(dx,depth,dt,years,c,shift=3.):
    """
    Function runs the same solver but adding a uniform shift
    to the temperatures (in degrees C)

    Parameters
    ----------
    temps : callable
        Function to determine the temperatures, defaults to 
        kanger_bounds
    shift : float
        uniform shift to add to the initial conditions
    """
    # define bounds function to allow shift
    def shift_bounds(u_arr,time_arr):
       return kanger_bounds(u_arr,time_arr,shift)
    
    # run with everything, including debugging to check it works
    x_kang,t_kang,u_kang = heatdiff(xmax=depth,tmax=years,dx=dx,dt=dt,
                                    bconds=shift_bounds,c2=c,init=0,debug=0)
    return x_kang,t_kang,u_kang
    
def shift_temps():
    """
    solve for all the warming scenarios and then plot heatmaps of
    individual and combined profiles of temperature.

    This function will produce the plots necessary for every question
    as there is no point in running redundant code.

    """
    # depth is 0m to 100m with dx of 1
    dx    = 0.5             # unit of m
    depth = 100             # unit of m
    dt    = 1*86400        # unit of seconds
    years  = 365*60*86400   # unit of seconds
    c      = 2.5e-7         # converted value in lab to m^2/s
    
    # plot temperature profile
    fig, ax = plt.subplots(1,2,figsize=(12,8))
    us,ds = [],[]
    shifts=[3.,1.,.5,0.]
    for shift in shifts:
        fig2, ax2 = plt.subplots(1,2,figsize=(12,8))
        x,t,u = permafrost_solve(dx,depth,dt,years,c,shift)
        us.append(u)
        ds.append(x)
        fig1, _ = heatmap(u,t,x,'years')
        plot_all_profiles([u],[x],1*86400,ax2[0],[shift],colors=COLORS)
        plot_depth_thickness([u],[x],1*86400,ax2[1],[shift],colors=IBM_CMAP)

        fig1.savefig(f'{OUTPATH}/heatmap_{shift}.png',dpi=300)
        fig2.savefig(f'{OUTPATH}/temp_profile_{shift}.png',dpi=300)

    plot_all_profiles(us,ds,1*86400,ax[0],shifts,colors=COLORS[::-1])
    plot_depth_thickness(us,ds,1*86400,ax2[1],shifts,colors=IBM_CMAP)
    
    fig.savefig(f'{OUTPATH}/kang_warmed_heatmap.png',dpi=300)

class TestHeatdiff(unittest.TestCase):
    """
    Class contains unit tests for the verification step in the 
    lab methodology.

    Adapted from https://www.geeksforgeeks.org/unit-testing-python-unittest/
    """
    def test_heateqn(self):
        """
        Unit test solves for wire test case given in lab code. This has 
        initial conditions of 4*x - 4*(x**2) and boundary conditions set
        to 0C.

        Also creates and saves a heatmap of the solver solution and the
        reference solution to be shown in the lab.

        Parameters
        ----------
        self : Self@TestHeatdiff
            unit test class object
        """
        # Solution to problem 10.3 from fink/matthews as a nested list:
        sol10p3 = [[0.000000, 0.640000, 0.960000, 0.960000, 0.640000, 0.000000],
        [0.000000, 0.480000, 0.800000, 0.800000, 0.480000, 0.000000],
        [0.000000, 0.400000, 0.640000, 0.640000, 0.400000, 0.000000],
        [0.000000, 0.320000, 0.520000, 0.520000, 0.320000, 0.000000],
        [0.000000, 0.260000, 0.420000, 0.420000, 0.260000, 0.000000],
        [0.000000, 0.210000, 0.340000, 0.340000, 0.210000, 0.000000],
        [0.000000, 0.170000, 0.275000, 0.275000, 0.170000, 0.000000],
        [0.000000, 0.137500, 0.222500, 0.222500, 0.137500, 0.000000],
        [0.000000, 0.111250, 0.180000, 0.180000, 0.111250, 0.000000],
        [0.000000, 0.090000, 0.145625, 0.145625, 0.090000, 0.000000],
        [0.000000, 0.072812, 0.117813, 0.117813, 0.072812, 0.000000]]
        # Convert to an array and transpose it to get correct ordering:
        sol10p3 = np.array(sol10p3).transpose()

        # solve the heat equation for the test case
        x,t,solver_result = heatdiff(1,0.2,0.2,0.02)

        # confirm the coefficients and the temperatures are equal to 1 decimal place
        np.testing.assert_allclose(solver_result, sol10p3, rtol=1E-5)
        # print the date and time of the unit test for reference
        now = datetime.datetime.now()
        print(f'Unit test executed at {now.strftime("%Y-%m-%d %H:%M:%S")}')

def main():
    """
    main function executes all code necessary to reproduce
    the figures in the lab report. Will execute all code in order.
    """
    # create path for figures to be saved if not already existing
    global OUTPATH
    if not OUTPATH:
        # will create a figures directory in your current working directory
        OUTPATH = f'{os.getcwd()}/laratt_lab04_figures'

    Path(OUTPATH).mkdir(parents=True, exist_ok=True)
    shift_temps()

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