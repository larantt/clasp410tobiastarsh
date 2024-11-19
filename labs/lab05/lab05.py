#!/usr/bin/envs python3
"""
lab05.py - snowball earth
Author : Lara Tobias-Tarsh (laratt@umich.edu)

"""
import numpy as np
import matplotlib.pyplot as plt

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

# Defin
IBM_BLUE = "#648FFF"
IBM_PURPLE = "#785EF0"
IBM_PINK = "#DC267F"
IBM_ORANGE = "#FE6100"
IBM_YELLOW = "#FFB000"

# set lambda (total diffusion coeffs)
LAMBDA_DEFAULT = 100.  # m^2/s, D/rho * C, D is heat diffusion coeff

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
    lats_in : Numpy array
    Array of latitudes in degrees where temperature is required.
    0 corresponds to the south pole, 180 to the north.
    Returns
    -------
    temp : Numpy array
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


def snowball_earth(nbins=18, dt=1, tstop=10000, debug=0, lam=100., 
                   spherecorr = True, initialise_only=False, albedo=0.3, emiss=1., s0=1370.):
    """

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
    # create initial condition
    temp = temp_warm(lats)
    # number of timesteps
    nstep = int(tstop/dt)

    if initialise_only:
        return lats, temp

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
    for i in range(nstep):
        # add spherical correction term
        if spherecorr:
            temp += dt_sec * lam * dA_xz * np.matmul(B_matrix, temp)
        
            radiative = (1-albedo) * insolation(s0,lats)  - emiss*SIGMA*(temp+273.15)**4
            temp += dt_sec * radiative / (RHO*C*MXDLYR)
        
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

    Parameters
    ----------

    Returns
    -------
    """
    # Generate very simple grid
    lats, initial = snowball_earth(initialise_only=True)
    lats2, t_diff = snowball_earth(spherecorr=False)
    lats3, t_sphere = snowball_earth()

    fig,ax = plt.subplots(1,1,figsize=(12,8))
    ax.plot(lats,initial,label='Warm Earth Initial Conditions',lw=3, c=IBM_BLUE)
    ax.plot(lats2,t_diff, label='Simple heat diffusion',lw=3, c=IBM_PINK)
    ax.plot(lats3,t_sphere, label='Simple heat diffusion + Spherical Correction',lw=3, c=IBM_PURPLE)

    ax.set_xlabel('Latitude ($^{\\circ}$ (0 = South Pole))',fontweight='bold')
    ax.set_ylabel('Temperature  ($^{\\circ} C$)',fontweight='bold')

    ax.legend(frameon=False,ncols=2)

