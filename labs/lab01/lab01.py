#!/usr/bin/env python3
"""
lab01.py

Author : Lara Tobias-Tarsh
Last Modified : 9/16/24

The lab demonstrates how the principle of universality can be used to simulate a 
number of scenarios, in this case the spread of forest fires or diseases. The model
implemented in this script represents the fire or disease as an nx by ny rectangular grid.

In each simulation, the grid is initialised by either starting a single cell, or starting
a random number of cells bare and on fire, depending on user input. To allow as much reproducability
as possible, the initial conditions can be passed a value to initiate a predefined random number generator.
The fire spread itself is more stochastic, and is does not have the same pseudo-stochatsic charateristics.

Running this file will create 4 directories containing figures for the simulation with the following figures:
    * q1_3x3_forest
        - Matrix plots of the fire at each step for a 3 x 3 grid (Figure 1 in report)
    
    * q1_3x5_forest
        - Matrix plots of the fire at each step for a 3 x 5 grid (Figure 2 in report)
    
    * q2_500x250_disease
        - Heatmap panels describing the evolution of a disease for a 
        500 x 250 grid, varying the mortality rate and early vaccine rate (Figures 4 & 6 in report)
        - Twin axes plots describing the final state of the population for each experiment, both varying 
        the mortality rate and early vaccine rate (Figures 3 & 5 in the report)
    
    * q2_500x250_forest
        - Heatmap panels describing the evolution of a wildfire for a 
        500 x 250 grid, varying the probability of spread and initial forest density (Figures 8 & 10 in report) 
        - Twin axes plots describing the final state of the population for 
        each experiment, both varying the mortality rate and early vaccine rate (Figures 7 & 9 in report)

To execute the experiments in file, which correspond to the questions in Lab 1, simply run:

```
python3 lab01.py
```
"""
#############
## IMPORTS ##
#############
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
plt.style.use('seaborn-v0_8-talk')

#############
## GLOBALS ##
#############
# define location to save figures
OUTPATH = '/Users/laratobias-tarsh/Documents/fa24/clasp410tobiastarsh/labs/lab01/figures'

# STATE VARIABLES FOR EACH CELL
DEAD = 0        # cell dies, always constant
IMMUNE = 1      # cell is immune, or corresponds to a bare forest cell
HEALTHY = 2     # cell is healthy and not immune, or forested
SPREADING = 3   # cell is infected with disease or on fire (it can spread)

# set colormaps, chosen to be somewhat colorblind frindly
DISEASE_CMAP = {
        'Dead' : '#6C6061',
        'Immune' : '#F1ECCE',
        'Healthy' : '#519872',
        'Sick'  : '#F08A4B'
}
FOREST_CMAP = {
        'Bare' : '#F1ECCE',
        'Forested' : '#519872',
        'Burning'  : '#F08A4B'
}

###############
## FUNCTIONS ##
###############

def progress_heatmaps(hmap_arr,p_spreads,run_name,ext,title,ylab='spread',colormap='Greys',vmax=1.0):
    """
    Helper function to plot and format each panel for the
    progress heatmaps shown in the report for questions 2 and 3.
    These are inspired by hovmoller diagrams commonly used in meteorology.

    NOTE: these heatmaps will not be perfectly reproducible, as they
    describe the evolution of each state variable with time. As a result
    they are significantly more impacted by the pseudo-stochastic nature
    of the model than the summary plots that analyse the final state of 
    the model grid. 
    
    This is especially relevant for "flare ups", where a
    the fraction of cells in a given state falls below the discernable
    range of the colorbar for a number of iterations, as the fraction of
    the grid in that state is extremely small, however this number then
    increases briefly later in time, causing it to appear as a blip like
    feature on the heatmap. In these cases, there is always at least one
    cell in the state variable, but the fractions are so small that adjusting
    the colorbar to capture ALL miniscule fluctuations in the proportion of a 
    large grid in a given state would be near impossible.

    Getting these to format with gridspec was a nightmare, so I 
    decided to just make single panel plots for my own sanity.

    Parameters
    ----------
    hmap_arr : np.ndarray
        multidimensional array containing the simulation progress
        this is created in the question 2 and 3 functions.
    p_spreads : arrayLike
        spread probabilities used to label the axes
    run_name : str
        path to which to save each figure panel
    ext : str
        string used to save each panel
    title : str
        title for plot
    ylab : str
        probability being varied for ylabel
    colormap : str or LinearSegmentedColormap
        matplotlib colormap to use in figure
    vmax : float
        max of plot (used if hard to read)

    Returns
    -------
    None
    """
    # FIGURE 3: heatmaps for each timestep and probability of spread
    fig, ax = plt.subplots(1,1,figsize=(10,6))
    # imshow data, scaling to fit the figure   
    contour = ax.imshow(hmap_arr,cmap=colormap,vmax=vmax,vmin=0.0,aspect='auto')
    # create colorbar
    cbar = plt.colorbar(contour,ax=ax,orientation='vertical',pad=0.01)
    cbar.set_label(label='Proportion of Total Grid Cells',weight='bold',labelpad=-20)
    # set the colorbar labels from 0 to max, not intermediate ticks
    cbar.set_ticks((0.0,vmax),labels=('0.0',f'{vmax:.3f}'))

    # set y ticks
    ax.set_yticks(np.arange(0,11,1),labels=p_spreads)

    # set axes labels
    ax.set_ylabel(f'{ylab.title()}',fontsize=12,fontweight='bold')
    ax.set_xlabel('Number of Iterations',fontsize=12,fontweight='bold')

    # loop through and make axes bold and pretty
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2)
    # make labels bold
    # increase tick width
    ax.tick_params(width=2)

    # set title
    ax.set_title(title,fontsize=16,fontweight='bold',loc='left')
    # set tight layout
    fig.tight_layout()
    fig.savefig(f'{OUTPATH}/{run_name}/{ext}.png')

def format_twin_axes(axis,twin_axis,lines,ylabels,xlabel,left_title=None,right_title=None):
    """
    Helper function to nicely format twin axes and avoid lots of
    ugly boilerplate code in the plotting sections. Designed to be
    portable across many applications.

    Does not support more than one axes twin because i am morally
    opposed to doing this and it is not readable or interpretable.

    Note, contains a local function to make extra axes invisible called
    make_patch_spines_invisible(). This is documented in the local function.

    Inspiration for this function is from this Stack Exchange post:
    https://stackoverflow.com/questions/58337823/matplotlib-how-to-display-and-position-2-yaxis-both-on-the-left-side-of-my-grap

    Parameters
    ----------
    axis : mpl.axes
        main axes object for figure
    twin_axis : mpl.axes
        axes object with twin y axis
    ylabels : list(str)
        list of labels to assign to each axis twin
    xlabel : str
        label for the common x axis
    left_title : str
        main title for the figure
    right_title : str
        additional title formatted on the right of the figure
        used here for listing fixed probabilities in the simulation.

    Returns
    -------
    None

    """
    def make_patch_spines_invisible(ax):
        """
        Local helper function to make extra axes spines invisible and give
        more control over formatting.

        Parameters
        ----------
        ax : mp.axes
            axis to hide spines on
        """
        ax.set_frame_on(True)
        ax.patch.set_visible(False)
        for sp in ax.spines.values():
            sp.set_visible(False)

    labkw = dict(size=14,weight='bold') # dictionary formats the axes labels to look nice
    tkw = dict(size=4, width=3) # dictionary formats the tick labels to look good

    # first hide all the spines on each axis so they can be manually recreated more nicely
    make_patch_spines_invisible(axis)
    make_patch_spines_invisible(twin_axis)

    # move the y twin next to the original y axis
    axis.spines["left"].set_position(("axes", -0.01))
    twin_axis.spines["left"].set_position(("axes", -0.1))

    # turn the original bottom axes back on
    axis.spines['bottom'].set_visible(True)
    # make spines bold because it looks pretty
    axis.spines['bottom'].set_linewidth(3)
    # make spines grey because it looks pretty
    tkw = dict(size=4, width=3) # dictionary formats the tick labels to look good
    axis.tick_params(axis='x', colors="#3b3b3b", **tkw)
    axis.spines['bottom'].set_color("#3b3b3b")
    axis.xaxis.label.set_color("#3b3b3b")
    # set the xlabel
    axis.set_xlabel(f'{xlabel}',c="#3b3b3b",**labkw)


    # now, loop over the axes and format nicely
    for ax,line,ylab in zip([axis,twin_axis],lines,ylabels):
        # turn on the axes as appropriate
        ax.spines["left"].set_visible(True)
        ax.yaxis.set_label_position('left')
        ax.yaxis.set_ticks_position('left')

        # make the axes thicker bc it looks pretty
        ax.spines['left'].set_linewidth(3)

        # set the colors and ticks to match the lines
        ax.spines['left'].set_color(line.get_color())
        ax.yaxis.label.set_color(line.get_color())
        ax.tick_params(axis='y', colors=line.get_color(), **tkw)

        # set the x label and y label to match the lines
        ax.set_ylabel(ylab,c=line.get_color(), **labkw)
        
        # force axes to start at 0
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

    # set titles
    if left_title:
        axis.set_title(f'{left_title}',fontsize=19,fontweight='bold',loc='left',x=-0.1,y=1.07)
    if right_title:
        axis.set_title(f'{right_title}',fontsize=18,fontweight='bold',loc='right',y=1.07,c="#3b3b3b")

    # plot legend
    legend_props = {"size" : 14, "weight" : 'bold'}
    axis.legend(lines, [l.get_label() for l in lines], frameon=False, labelcolor='#3b3b3b',
                loc='upper left',prop=legend_props,bbox_to_anchor=(-0.12,1.079),ncols=2)
        
def spread(matrix, p_spread, p_death):
    """
    function finds the indices of all adjacent cells to the burning/infectious cell.

    Uses concept of universality again through the decide_fate() function which can
    be used to either kill/immunise a cell that is spreading, or infect/not infect
    a neighbouring cell.

    Accesses state variables at the top of the file to set values.

    Parameters
    ----------
    matrix : np.ndarray
        multidimensional array representing the model grid
    p_spread : float
        probability that the fire/disease will spread to a neighbour
    p_death : float
        probability that a burning/infected cell will die

    Returns
    -------
    next_matrix : np.ndarray
        multidimensional array representing the model grid at the next state
     
    """
    next_matrix = matrix.copy() # copy the current state so this can be edited
    
    # get the size of the matrix to check boundary conditions
    m, n= np.array(matrix.shape)

    # loop over cells which are "spreaders"
    for i,j in np.argwhere(matrix==SPREADING):
        # if the position is not on the border, and it is a tree
        if i > 0:
            if matrix[i-1,j] == HEALTHY:
                # use p_die parameter to roll dice for probability of spread, immune to not burn, dead to burn
                next_matrix[i-1,j] = SPREADING if np.random.rand() < p_spread else HEALTHY
        
        # if the position is not on the border, and it is a tree
        if i+1 < m:
            if matrix[i+1,j] == HEALTHY:
                # assign copy spreader value
                next_matrix[i+1,j] = SPREADING if np.random.rand() < p_spread else HEALTHY
        
        # if the position is not on the border, and it is a tree
        if j > 0:
            if matrix[i,j-1] == HEALTHY:
                # use p_die parameter to roll dice for probability of spread, immune to not burn, dead to burn
                next_matrix[i,j-1] = SPREADING if np.random.rand() < p_spread else HEALTHY
        
        # if the position is not on the border, and it is a tree
        if j+1 < n:
            if matrix[i,j+1] == HEALTHY:
                # use p_die parameter to roll dice for probability of spread, immune to not burn, dead to burn
                next_matrix[i,j+1] = SPREADING if np.random.rand() < p_spread else HEALTHY
    
        # finally, decide fate of the spreader cell
        next_matrix[i,j] = DEAD if np.random.rand() < p_death else IMMUNE
    # return new state of fire
    return next_matrix

def full_simulation(matrix, p_spread, p_death=0.0, vis=None, cmap=FOREST_CMAP):
    """
    Executes a full simulation of a forest fire/disease event by
    running a while loop to check if any burning/sick cells
    exist in the simulation, executing a spread sequence if they
    do. 
    
    The proportion of cells in each state are calculated at
    the end of each iteration and saved in a list, which is 
    more memory efficient than storing the entire grid, though
    for verification purposes, each step can be visualised  and 
    saved to disk desired.

    Parameters
    ----------
    matrix : np.ndarray
        2D array describing the initial state of the simulation grid
    p_spread : float
        probability the fire will spread
    p_death : float
        probability an infected cell will die
        defaults to 0.0 (aka wildfire configuration)
    vis : str
        if visualisations of the grid are desired, pass 
        the run name to save the file under.
    cmap : dict
        colormap for visualising the data if vis is passed
        to the function.

    Return
    ------
    state_0 : list
        proportion of grid with state variable DEAD at each step
        note this list is empty and redundant in forest fire configuration
    state_1 : list
        proportion of grid with state variable IMMUNE at each step
    state_2 : list
        proportion of grid with state variable HEALTHY at each step
    state_3 : list
        proportion of grid with state variable SPREADING at each step
    """
    # set up a counter to check iterations until break condition is met
    # note that because the simulation is initialised externally, counter
    # starts at 1, as t=0 has already passed.
    count = 1

    # initialise lists to store proportion of cells in each state
    # note that for the forest fire configuration, state_0 will be an empty list
    state_0, state_1, state_2, state_3 = [], [], [], []
    # now, run while loop until condition to break (no burning cells) is met
    while SPREADING in matrix:
        matrix = spread(matrix,p_spread,p_death)
        # if visualisation exists, run visualise the model at each iteration
        if vis:
            # ensure a colormap exists
            visualise_current_state(matrix,f'Step: {count}',vis,f'unit_test_step-{count:.1f}',cmap)

        # calculate and save proportion of the cells in each state for plotting
        state_0.append((np.size(matrix[matrix == DEAD])/(np.size(matrix))))
                    
        state_1.append((np.size(matrix[matrix == IMMUNE])/(np.size(matrix))))

        state_2.append((np.size(matrix[matrix == HEALTHY])/(np.size(matrix))))   
                    
        state_3.append((np.size(matrix[matrix == SPREADING])/(np.size(matrix))))

        # iterate counter
        count +=1
    
    return state_0, state_1, state_2, state_3

def initialise_simulation(ni,nj,p_start,p_bare=0.0,generator=None):
    """
    Function initialises forest fire/disease spread simulation based on a set of 
    probabilities for the initial state of the grid.

    Accesses global state variables to initialise the simulation.

    Parameters
    ----------
    ni : int
        size of matrix in the x dimension
    nj : int
        size in the y dimension
    p_start : float or Tuple
        probability that the cell starts as a spreader (e.g. fire/sick).
        if tuple, fire is started only at that location.
    p_bare : float
        probability that the cell starts bare/immune
        defaults to 0 (e.g. no bare spots)
    generator : int
        integer to initialise a random generator to 
        ensure repeatable initial conditions
    Returns
    -------
    initial_matrix : np.ndarray
        2D array with initial state of the simulation

    """
    # generate healthy forest/population
    initial_matrix = np.zeros((ni,nj),dtype=int) + HEALTHY

    # create bare spots/immunity
    if isinstance(generator,int):
        # create and pass a random generator to ensure initial conditions are always the same
        # note that to prevent these probabilities being identical to the start probabilites
        # the generator is incremented by 5.
        rng = np.random.default_rng(generator+5)
        # set some "random" cells to bare
        isbare = rng.random((ni,nj)) < p_bare
        initial_matrix[isbare] = IMMUNE
    else:
        # no generator, just start randomly based on p_bare
        isbare = np.random.rand(ni,nj) < p_bare
        initial_matrix[isbare] = IMMUNE
    
    # start some fires
    if isinstance(p_start,tuple):
        # index at specified tuple index to start fire
        initial_matrix[p_start] = SPREADING
    else:
        if isinstance(generator,int):
            # create and pass a random generator to ensure initial conditions are always the same
            rng = np.random.default_rng(generator)
            # start some "random" fires/infect some randos
            start = rng.random((ni,nj)) < p_start
            initial_matrix[start] = SPREADING
        else:
            # no generator, just start randomly based on p_start
            start = np.random.random((ni,nj)) < p_start
            initial_matrix[start] = SPREADING

    
    # return initial state of the simulation
    return initial_matrix

def visualise_current_state(matrix,title,run_name,exten,cmap):
    """
    Function visualises the forest fire spread, mostly
    used for testing purposes, and writes it to file.

    Parameters
    ----------
    matrix : np.ndarray
        matrix at a given step in the simulation
    title : str
        string to use as title for the figure
    run_name : str
        used to construct the path to the file
    exten : str
        extension used in filename

    Returns
    -------
    None
    """
    # Generate our custom segmented color map for this project (chosen to be colorblind friendly).
    colormap = ListedColormap([val for val in cmap.values()])
    fig, ax = plt.subplots(1, 1,figsize=(12,8)) # set up the figure
    # NOTE: need to fix such that range automatically adjusts on colorbar
    contour = ax.matshow(matrix, vmin=1, vmax=3,cmap=colormap) # plot the forest
    
    ax.set_title(f'{title}',loc='left',fontsize=16,fontweight='bold',y=1.05) # set title
    ax.set_xlabel('X (km)')  # Label X/Y assuming km.
    ax.xaxis.set_ticks_position('bottom') # move xticks to bottom
    ax.set_ylabel('Y (km)')


    # spoof elements for a legend
    legend_elements = [Patch(facecolor=value, edgecolor='k', label = key) for key, value in cmap.items()]
    ax.legend(handles=legend_elements, ncols=4,frameon=False,loc='upper left',bbox_to_anchor=(-0.02,1.07),fontsize=14)

    # make directory to save plots if it doesn't already exist
    Path(f'{OUTPATH}/{run_name}').mkdir(parents=True, exist_ok=True)
    # save the figure at each iteration
    fig.savefig(f'{OUTPATH}/{run_name}/{exten}.png')

def question_one(nx,ny):
    """
    Simulation for question 1, which is a 3x3 test forest

    This utilises the "testing" parameter in initialsie_simulation() to
    start the fire at the center cell

    Parameters
    ----------
    nx : int
        number of cells in the simulation grid in x-direction
    ny : int
        number of cells in the simulation grid in y direction

    Returns
    --------
    None

    Figures
    --------
    visualises the simulation in individual panels at each iteration.
    """
    # set path to save file
    run_name = f'q1_{nx}x{ny}_forest' # directory name to save figures

    # define probability of spread
    prob_spread=1.0

    # initialise forest, start fire at middle location
    start = (nx//2,ny//2)
    current_state = initialise_simulation(nx,ny,start)
    
    visualise_current_state(current_state,'Step: 0',run_name,'unit_test_step-0',FOREST_CMAP)
    # run forest fire simulation. 
    # do not store output (proprotion of cells in each state) as is not needed so use _
    _ = full_simulation(current_state,prob_spread,vis=run_name)

def question_two(nx,ny,fixed_prob=0.0,prob_to_vary='spread'):
    """
    Function to solve the simulations for Question 2: how does the spread of wildfire depend on the
    probability of spread of fire and initial forest density?. 

    In order to do this, we vary either the probability of the fire spreading, or the probability
    that a given cell is bare at the start of the simulation by iterating over an array of 
    probabilities for either p_bare or p_spread.

    Parameters
    ----------
    nx : int
        number of cells in the x direction to initialise the matrix
    ny : int
        number of cells in the y direction to initialise the matrix
    fixed_prob : float
        the fixed probability to use
        Defaults to 1.0, assigned to the variable not being changed
    prob_to_vary : string
        name of the probability to vary 
        Defaults to 'spread', but can be either 'spread' or 'bare'

    Returns
    -------
    None

    Figures
    -------    
    Generates dual axes plots of the forest evolution with time and 
    generates heatmaps to explore timeseries evolution of each simulation with
    varying probbabilities.
    """
    # Set globals for this run
    p_variable = np.arange(0.0,1.1,0.1) # array of probabilities to iterate over
    prob_start = 0.01 # start fire based on pseudo-random seeding probabilities
    # arbitrary integer passed to random number generator, used to ensure repeatable initial conditions
    gen_int = 2021 
    
    run_name = f'q2_{nx}x{ny}_forest' # directory name to save figures
    # make directory to save plots if it doesn't already exist
    Path(f'{OUTPATH}/{run_name}').mkdir(parents=True, exist_ok=True)
    
    # initialise empty dictionaries for forested, bare and burned
    forested_dict, bare_dict, burning_dict = {}, {}, {}

    # catch exceptions to the prob to vary function - allows simple if else syntax later on
    if prob_to_vary not in {'spread','bare'}:
        raise(ValueError(f'prob_to_vary: {prob_to_vary} is invalid. Choose from "spread" or "bare"'))

    # iterate over the variable array of probabilities
    for vary in p_variable:
        if prob_to_vary == 'spread':
            # keep initial conditions constant with fixed_prob    
            current_state = initialise_simulation(nx,ny,prob_start,fixed_prob,generator=gen_int)
            # vary the probability of spread
            # underscore used because there is no need to save proportion dead for wildfire
            _, prop_bare, prop_forest, prop_burn = full_simulation(current_state,vary)
        else:
            # vary initial conditions (proportion of forest burned) using vary
            current_state = initialise_simulation(nx,ny,prob_start,vary,generator=gen_int)
            # hold the probability of spread constant
            # underscore used because there is no need to save proportion dead for wildfire
            _, prop_bare, prop_forest, prop_burn = full_simulation(current_state,fixed_prob)

        # save to dictionaries for easy access later in plotting sequence            
        forested_dict[f'{vary:.1f}'] = prop_forest # key indicates probability of spread
        bare_dict[f'{vary:.1f}'] = prop_bare # key indicates probability of spread
        burning_dict[f'{vary:.1f}'] = prop_burn # key indicates probability of spread
    
    #----------#    
    # PLOTTING #
    #----------#
    # calculate important quantities for plotting
    p_spreads = np.array(list(forested_dict.keys()),dtype=float)
    niters = [len(value) for key,value in forested_dict.items()] # assume all are same length
    # calculate the proportion of the forest burned at the end (final bare - inital bare)
    prop_burned = [(value[-1]-value[0]) for key,value in bare_dict.items()]
    init_bare = [value[0] for key,value in bare_dict.items()]

    if prob_to_vary == 'spread':
        # FIGURE 1: Probability the fire will spread/starts bare vs the number of iterations
        # for the fire to burn out fully
        fig1, ax1 = plt.subplots(1,1,figsize=(12,8))  # set up figure and axes

        p1 = ax1.plot(p_spreads,niters,'-o',label='Time for Fire to Spread',c='#6F8FAF',lw=3)
        
        # set up second, dual y axis
        ax2 = ax1.twinx()
        p2 = ax2.plot(p_spreads,prop_burned,'-o',label='Fraction Burned',c="#CC8899",lw=3)

        # labels to pass to formatting function
        ax1_xlab = 'Probability of Fire Spreading'
        ax1_ylab = 'Number of Iterations'
        ax2_ylab = 'Fraction of Forest Burned'
        ax_title_left = 'Probability of Spread Impact on Wildfire Evolution'
        ax_title_right = f'p_bare: {fixed_prob}'

        # call format function to make it look pretty
        format_twin_axes(ax1,ax2,[p1[0],p2[0]],[ax1_ylab,ax2_ylab],ax1_xlab,ax_title_left,ax_title_right)
        
        # make everything fit on plot nicely
        fig1.tight_layout()
        fig1.savefig(f'{OUTPATH}/{run_name}/prob_spread_vs_niters.png')
    else:
        fig1, ax1 = plt.subplots(1,1,figsize=(12,8))  # set up figure and axes
        p1 = ax1.plot(init_bare,niters,'-o',label='Time for Fire to Spread',c='#6F8FAF',lw=3)
        
        # create twin axes to show additional info on same plot
        ax2 = ax1.twinx()
        p2 = ax2.plot(init_bare,prop_burned,'-o',label='Fraction Burned',c="#CC8899",lw=3)

        # labels to pass to formatting function
        ax1_xlab = 'Fraction of Forest Bare at Initialization'
        ax1_ylab = 'Number of Iterations'
        ax2_ylab = 'Fraction of Forest Burned'
        ax_title_left = 'Initial Forest Density Impact on Wildfire Spread'
        ax_title_right = f'p_spread: {fixed_prob}'

        # call format function to make it look pretty
        format_twin_axes(ax1,ax2,[p1[0],p2[0]],[ax1_ylab,ax2_ylab],ax1_xlab,ax_title_left,ax_title_right)

        # make everything fit on plot nicely
        fig1.tight_layout()
        fig1.savefig(f'{OUTPATH}/{run_name}/prob_bare_vs_niters.png')
    

    # FIGURE 3: heatmaps for each timestep and probability of spread
    # NOTE: need to improve formatting and visuals
    fig3, ax3 = plt.subplots(3,1,figsize=(8,14),sharey=True)
    # construct heatmap shapes
    hmap_length = np.max(niters)  # longest number of iterations wide
    hmap_width = len(p_spreads) # one row for each probability of spread

    # initialise 3D array to store proportions for each state
    hmap_arr = np.zeros((3,hmap_length,hmap_width),dtype=float)
    # populate hmap_arr with probabilities
    for row,(forest,bare,burn) in enumerate(zip(forested_dict.values(),
                                                bare_dict.values(),burning_dict.values())):
        
        hmap_arr[0,:len(forest),row] = forest # fill the forest part of the array
        hmap_arr[1,:len(bare),row] = bare # fill the bare part of the array
        hmap_arr[2,:len(burn),row] = burn # fill the burning part of the array

    # heatmaps for each timestep and probability of spread
    # generate colormap corresponding to color used in line plot
    # Edit this gradient at https://eltos.github.io/gradient/#FFFFFF-648FFF (cool site!)
    cmap_forest = LinearSegmentedColormap.from_list('forested',(
    (0.000, (1.000, 1.000, 1.000)),
    (1.000, (0.271, 0.506, 0.380))))
    progress_heatmaps(hmap_arr[0,:,:].T,p_spreads,run_name,f'{prob_to_vary}_heatmap_forest',
                      'Proportion Forested',ax1_xlab,cmap_forest)
    
    cmap_bare = LinearSegmentedColormap.from_list('bare', (
    (0.000, (1.000, 1.000, 1.000)),
    (1.000, (0.682, 0.643, 0.400))))
    progress_heatmaps(hmap_arr[1,:,:].T,p_spreads,run_name,f'{prob_to_vary}_heatmap_bare',
                      'Proportion Bare',ax1_xlab,cmap_bare)
    
    cmap_burn = LinearSegmentedColormap.from_list('burning', (
    (0.000, (1.000, 1.000, 1.000)),
    (0.0001, (1.000, 0.922, 0.847)), # trying to help show very small proportion infected
    (1.000, (0.996, 0.380, 0.000))))
    progress_heatmaps(hmap_arr[2,:,:].T,p_spreads,run_name,f'{prob_to_vary}_heatmap_burn',
                      'Proportion Burning',ax1_xlab,cmap_burn,vmax=0.005)
    
def question_three(nx,ny,fixed_prob=0.0,prob_to_vary='death',prob_spread=1.0):
    """
    Function to solve the simulations for Question 2: how does the spread of wildfire depend on the
    probability of spread of fire and initial forest density?. 

    In order to do this, we vary either the probability of the fire spreading, or the probability
    that a given cell is bare at the start of the simulation by iterating over an array of 
    probabilities for either p_bare or p_spread.

    Parameters
    ----------
    nx : int
        number of cells in the x direction to initialise the matrix
    ny : int
        number of cells in the y direction to initialise the matrix
    fixed_prob : float
        the fixed probability to use
        Defaults to 1.0, assigned to the variable not being changed
    prob_to_vary : string
        name of the probability to vary 
        Defaults to 'spread', but can be either 'spread' or 'bare'

    Returns
    -------
    None

    Figures
    -------    
    Generates dual axes plots of the forest evolution with time and 
    generates heatmaps to explore timeseries evolution of each simulation with
    varying probbabilities.
    """
    
    # Set globals for this run
    p_variable = np.arange(0.0,1.1,0.1) # array of probabilities to iterate over
    prob_start = 0.01 # start fire based on pseudo-random seeding probabilities
    prob_spread = 1.0 # set probability of spread at arbitrary 1.0 to eliminate noise
    gen_int = 2021 # arbitrary integer to initialise numpy's random number generator
    
    run_name = f'q2_{nx}x{ny}_disease' # directory name to save figures
    # make directory to save plots if it doesn't already exist
    Path(f'{OUTPATH}/{run_name}').mkdir(parents=True, exist_ok=True)
    
    # initialise empty dictionaries for forested, bare and burned
    dead_dict, immune_dict, healthy_dict, spreading_dict = {}, {}, {}, {}

    # catch exceptions to the prob to vary function
    if prob_to_vary not in {'death','vaccination'}:
        raise(ValueError(f'prob_to_vary: {prob_to_vary} is invalid. Choose from death or vaccination'))

    # iterate over the array of probabilities
    for vary in p_variable:
        if prob_to_vary == 'death':
            # keep initial conditions constant with fixed_prob    
            current_state = initialise_simulation(nx,ny,prob_start,fixed_prob,generator=gen_int)
            # vary the mortality rate
            dead_list, immune_list, healthy_list, sick_list = full_simulation(current_state,prob_spread,vary)
        else:
            # vary initial conditions (early vax rate)
            current_state = initialise_simulation(nx,ny,prob_start,vary,generator=gen_int)
            # keep death rate constant with fixed_prob
            dead_list, immune_list, healthy_list, sick_list = full_simulation(current_state,prob_spread,fixed_prob)   
                    
        dead_dict[f'{vary:.1f}'] = dead_list # key indicates probability of spread
        immune_dict[f'{vary:.1f}'] = immune_list # key indicates probability of spread
        spreading_dict[f'{vary:.1f}'] = sick_list # key indicates probability of spread
        healthy_dict[f'{vary:.1f}'] = healthy_list # key indicates probability of spread 

    #----------#
    # PLOTTING #
    # ---------#
    tkw = dict(size=4, width=3) # dictionary formats the tick labels to look good
    legend_props = {"size" : 14, "weight" : 'bold'} # dictionary to format the legends

    p_spreads = np.array(list(dead_dict.keys()),dtype=float)
    # find the number of iterations for each simulation at its' probability of spread
    niters = [len(value) for key,value in dead_dict.items()] # assume all are same length

    # calculate the proportion of the population dead at end
    prop_dead = [value[-1] for key,value in dead_dict.items()]
    # calculate proportion immune at end
    prop_immune = [value[-1] for key,value in immune_dict.items()]
    prop_alive = [value[-1] for key,value in healthy_dict.items()]
    survival_rate = [sum(nums) for nums in zip(prop_immune,prop_alive)]
    
    fig1, (ax1,ax2) = plt.subplots(2,1,figsize=(12,10),sharex=True)           # set up figure and axes
    # now plot duration on line plot
    ax1.plot(p_spreads,niters,'-o',label='Disease Duration',c="#FFB000",lw=3)
    ax1.legend(frameon=False, labelcolor='#3b3b3b',
                loc='upper left',prop=legend_props,bbox_to_anchor=(-0.09,1.147),ncols=2)
    # now plot each state at end of simulation on line plot
    ax2.plot(p_spreads,prop_alive,'-o',label='No Immunity',lw=3,c="#DC267F")
    ax2.plot(p_spreads,prop_immune,'-o',label='Immune',lw=3,c="#785EF0")
    ax2.plot(p_spreads,prop_dead,'-o',label='Dead',lw=3,c="#648FFF")
    ax2.plot(p_spreads,survival_rate,'--',label='Total Surviving',lw=3,c="#FE6100")
    ax2.legend(frameon=False, labelcolor='#3b3b3b',
                loc='upper left',prop=legend_props,bbox_to_anchor=(-0.09,1.147),ncols=4)


    if prob_to_vary == "death":
        # label the axes
        ax1.set_ylabel('Number of Iterations',fontsize=14,fontweight='bold',c='#3b3b3b')
        # label axes 
        ax2_xlabel = 'Mortality Rate' # save to be used again in later heatmap plot
        ax2.set_xlabel(ax2_xlabel,fontsize=14,fontweight='bold',c='#3b3b3b')
        ax2.set_ylabel('Fraction of Population',fontsize=14,fontweight='bold',c='#3b3b3b')

        # set titles
        ax1.set_title('Impact of Mortality Rate on Disease Duration',
                      fontsize=20,fontweight='bold',loc='left',x=-0.08,y=1.1)
        ax1.set_title(f'p_vax = {fixed_prob:.1f}\np_spread = {prob_spread:.1f}',fontsize=18,
                      fontweight='bold',loc='right',y=1.1,c="#3b3b3b")
        ax2.set_title('Impact of Mortality Rate on Final Immunity',
                      fontsize=20,fontweight='bold',loc='left',x=-0.08,y=1.1)
        
    if prob_to_vary == "vaccination":
        # label the axes
        ax1.set_ylabel('Number of Iterations',fontsize=14,fontweight='bold',c='#3b3b3b')
        # label axes 
        ax2_xlabel = 'Vaccination Rate' # save to be used again in later heatmap plot
        ax2.set_xlabel(ax2_xlabel,fontsize=14,fontweight='bold',c='#3b3b3b')
        ax2.set_ylabel('Fraction of Population',fontsize=14,fontweight='bold',c='#3b3b3b')

        # set titles
        ax1.set_title('Impact of Early Vaccination Rate on Disease Duration',
                      fontsize=20,fontweight='bold',loc='left',x=-0.08,y=1.1)
        ax1.set_title(f'p_death = {fixed_prob:.1f}\np_spread = {prob_spread:.1f}',fontsize=18,
                      fontweight='bold',loc='right',y=1.1,c="#3b3b3b")
        ax2.set_title('Impact of Early Vaccination Rate on Final Immunity',
                      fontsize=20,fontweight='bold',loc='left',x=-0.08,y=1.1)


    # loop over axes to complete formatting
    for ax in [ax1,ax2]:
        # disconnect the axes slightly
        ax.spines["left"].set_position(("axes", -0.01))
        ax.spines["bottom"].set_position(("axes", -0.005))
        # make axes bold and pretty, set color to dark grey
        ax.spines["left"].set_linewidth(3)
        ax.spines["left"].set_color("#3b3b3b")

        # format y ticks
        ax.yaxis.label.set_color("#3b3b3b")
        ax.tick_params(axis='y', colors="#3b3b3b", **tkw)
        # format x ticks
        ax.tick_params(axis='x', colors="#3b3b3b", **tkw)
        ax.spines['bottom'].set_color("#3b3b3b")
        ax.xaxis.label.set_color("#3b3b3b")
        ax.spines["bottom"].set_linewidth(3)
        # hide unwanted axes
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # force axes to start at 0
        ax.set_xlim(left=0)
    
    # set different scales for readability
    ax1.set_ylim(bottom=np.min(niters)-1)
    ax2.set_ylim(bottom=0)

    fig1.tight_layout(pad=3.0) # ensure adequate spacing between plots
    # make directory to save plots if it doesn't already exist
    Path(f'{OUTPATH}/{run_name}').mkdir(parents=True, exist_ok=True)
    fig1.savefig(f'{OUTPATH}/{run_name}/figure1_{prob_to_vary}.png')   

    # construct heatmap shapes
    hmap_length = np.max(niters)  # longest number of iterations wide
    hmap_width = len(p_spreads) # one row for each probability of spread

    # initialise 3D array to store proportions for each state
    hmap_arr = np.zeros((4,hmap_length,hmap_width),dtype=float)
    # populate hmap_arr with probabilities
    for row,(dead,immune,healthy,sick) in enumerate(zip(dead_dict.values(), immune_dict.values(),
                                                healthy_dict.values(), spreading_dict.values())):
        
        hmap_arr[0,:len(dead),row] = dead # fill the dead part of the array
        hmap_arr[1,:len(immune),row] = immune # fill the immune part of the array
        hmap_arr[2,:len(healthy),row] = healthy # fill the healthy part of the array
        hmap_arr[3,:len(sick),row] = sick # fill the sick part of the array
    

    # FIGURE 3: heatmaps for each timestep and probability of spread
    # generate colormap corresponding to color used in line plot
    # Edit this gradient at https://eltos.github.io/gradient/#FFFFFF-648FFF (cool site!)
    cmap_dead = LinearSegmentedColormap.from_list('dead',(
    (0.000, (1.000, 1.000, 1.000)),
    (1.000, (0.392, 0.561, 1.000))))
    progress_heatmaps(hmap_arr[0,:,:].T,p_spreads,run_name,f'{prob_to_vary}_heatmap_dead',
                      'Proportion Dead',ax2_xlabel,cmap_dead)
    
    cmap_immune = LinearSegmentedColormap.from_list('immune', (
    (0.000, (1.000, 1.000, 1.000)),
    (1.000, (0.471, 0.369, 0.941))))
    progress_heatmaps(hmap_arr[1,:,:].T,p_spreads,run_name,f'{prob_to_vary}_heatmap_immune',
                      'Proportion Immune',ax2_xlabel,cmap_immune)
    
    cmap_alive = LinearSegmentedColormap.from_list('alive', (
    (0.000, (1.000, 1.000, 1.000)),
    (1.000, (0.863, 0.149, 0.498))))
    progress_heatmaps(hmap_arr[2,:,:].T,p_spreads,run_name,f'{prob_to_vary}_heatmap_healthy',
                      'Proportion No Immunity',ax2_xlabel,cmap_alive)
    
    if prob_to_vary == "death":
        # note that with this colormap, when almost 0% of the grid is infected the virus may still be
        # spreading but the heatmap will not be able to show this well. Account for this by varying scale.
        cmap_infected = LinearSegmentedColormap.from_list('infected', (
        (0.000, (1.000, 1.000, 1.000)),
        (0.0000001, (1.000, 0.922, 0.847)), # trying to help show very small proportion infected
        (1.000, (0.996, 0.380, 0.000))))
        progress_heatmaps(hmap_arr[3,:,:].T,p_spreads,run_name,f'{prob_to_vary}_heatmap_infected',
                        'Proportion Infected',ax2_xlabel,cmap_infected,vmax=0.05)
    else:
        cmap_infected = LinearSegmentedColormap.from_list('infected', (
        (0.000, (1.000, 1.000, 1.000)),
        (0.0000001, (1.000, 0.922, 0.847)), # trying to help show very small proportion infected
        (1.000, (0.996, 0.380, 0.000))))
        progress_heatmaps(hmap_arr[3,:,:].T,p_spreads,run_name,f'{prob_to_vary}_heatmap_infected',
                        'Proportion Infected',ax2_xlabel,cmap_infected,vmax=0.01)

      
# run simulations for question 1
question_one(3,3)
question_one(3,5)

# run simulations for question 2
question_two(500,250,fixed_prob=0.0,prob_to_vary='spread')
question_two(500,250,fixed_prob=1.0,prob_to_vary='bare')

# run simulations for question 3
question_three(500,250,fixed_prob=0.0,prob_to_vary='death')
# in this simulation, we fix the probability of death at a low, but not
# zero level in order to differentiate the dynamics of disease spread from the
# forest fire model.
question_three(500,250,fixed_prob=0.3,prob_to_vary='vaccination')

