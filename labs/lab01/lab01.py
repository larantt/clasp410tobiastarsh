#!/usr/bin/env python3
"""
lab01.py

Author : Lara Tobias-Tarsh
Last Modified : 9/11/24

The lab demonstrates how the principle of universality can be used to simulate a 
number of scenarios, in this case the spread of forest fires or diseases.

File contains the functions to execute a number of test simulations to prove this. To execute
this file, simply run:

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
plt.style.use('fivethirtyeight')

#############
## GLOBALS ##
#############
OUTPATH = '/Users/laratobias-tarsh/Documents/fa24/clasp410tobiastarsh/labs/lab01/figures'
# STATE VARIABLES FOR EACH CELL
DEAD = 0        # cell dies, always constant
IMMUNE = 1      # cell is immune, or corresponds to a bare forest cell
HEALTHY = 2     # cell is healthy and not immune, or forested
SPREADING = 3   # cell is infected with disease or on fire (it can spread)

###############
## FUNCTIONS ##
###############

def decide_fate(p_die,immune=IMMUNE,dead=DEAD):
    """
    Function decides the fate of a given cell in simulation

    Parameters
    ----------
    p_die : float
        probability of death between 0 and 1
    immune : int
        integer value indicating cell is immune in simulation
    dead : int
        integer value indicating cell is dead in simulation

    Returns
    -------
    immune : int
        integer value indicating cell is immune in simulation
    OR
    dead : int
        integer value indicating cell is dead in simulation
    """
    # roll the dice of life!
    if np.random.rand() < p_die: # random probability you live is below than death rate
        return dead # nooo... u died :(
    else:
        return immune # yay! you lived!

def spread(matrix, p_spread, p_death, avail=HEALTHY, spreader=SPREADING, immune=IMMUNE, dead=DEAD):
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
    for i,j in np.argwhere(matrix==spreader):
        # if the position is not on the border, and it is a tree
        if i > 0:
            if matrix[i-1,j] == avail:
                # use p_die parameter to roll dice for probability of spread, immune to not burn, dead to burn
                next_matrix[i-1,j] = decide_fate(p_die=p_spread,immune=avail,dead=spreader)
        
        # if the position is not on the border, and it is a tree
        if i+1 < m:
            if matrix[i+1,j] == avail:
                # assign copy spreader value
                next_matrix[i+1,j] = decide_fate(p_die=p_spread,immune=avail,dead=spreader)
        
        # if the position is not on the border, and it is a tree
        if j > 0:
            if matrix[i,j-1] == avail:
                # use p_die parameter to roll dice for probability of spread, immune to not burn, dead to burn
                next_matrix[i,j-1] = decide_fate(p_die=p_spread,immune=avail,dead=spreader)
        
        # if the position is not on the border, and it is a tree
        if j+1 < n:
            if matrix[i,j+1] == avail:
                # use p_die parameter to roll dice for probability of spread, immune to not burn, dead to burn
                next_matrix[i,j+1] = decide_fate(p_die=p_spread,immune=avail,dead=spreader)
    
        # finally, decide fate of the spreader cell
        next_matrix[i,j] = decide_fate(p_death,immune=immune,dead=dead)
        #print(f'Argwhere Spread Iteration Completed in {total_n}')
    # return new state of fire
    return next_matrix

def initialise_simulation(ni,nj,p_start,p_bare=0.0):
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
    p_start : float
        probability that the cell starts as a spreader (e.g. fire/sick)
    p_bare : float
        probability that the cell starts bare/immune
        defaults to 0 (e.g. no bare spots)
    Returns
    -------
    initial_matrix : np.ndarray
        2D array with initial state of the simulation

    """
    # generate healthy forest/population
    initial_matrix = np.zeros((ni,nj),dtype=int) + HEALTHY

    # create bare spots/immunity
    isbare = np.random.rand(ni,nj) < p_bare
    initial_matrix[isbare] = IMMUNE
    
    # start some fires
    if isinstance(p_start,tuple):
        # index at specified tuple index to start fire
        initial_matrix[p_start] = SPREADING
    else:
        # create and pass a random generator to ensure initial conditions are always the same
        rng = np.random.default_rng(2021)
        # start some "random" fires/infect some randos
        start = rng.random((ni,nj)) < p_start
        initial_matrix[start] = SPREADING
    
    # return initial state of the simulation
    return initial_matrix

def visualise_current_state(matrix,title,run_name,exten):
    """
    Function visualises the forest fire spread, mostly
    used for testing purposes, and writes it to file.

    Parameters
    ----------
    matrix : np.ndarray
        matrix at a given step in the simulation
    step : int
        step of the simulation that the matrix is on
    """
    # Generate our custom segmented color map for this project.
    forest_cmap = ListedColormap(['lightsteelblue','tan', 'darkgreen', 'crimson'])
    fig, ax = plt.subplots(1, 1) # set up the figure
    contour = ax.matshow(matrix, vmin=0, vmax=3,cmap=forest_cmap) # plot the forest
    ax.set_title(f'{title}',loc='left') # set title
    plt.colorbar(contour, ax=ax)

    # make directory to save plots if it doesn't already exist
    Path(f'{OUTPATH}/{run_name}').mkdir(parents=True, exist_ok=True)
    # save the figure at each iteration
    fig.savefig(f'{OUTPATH}/{run_name}/{exten}.png')

def q3_plots(results,run_name,var='dead'):
    """
    Function generates plots to summarise the progress of the simulation
    for use in questions 2 and 3

    Parameters
    ----------
    results : list(dict(list))
        dictionary of lists containing the proportions of each variable
        at each timestep
    run_name : string
        where to save the file
    """
    # figure 1 - how long for disease to stop spreading
    # NOTE: should probably add assertion to check if all are the same length
    fig1, ax1 = plt.subplots(1,1,figsize=(12,8))           # set up figure and axes

    # figure 2 - scatter plot of burned area at end and probability of spread with correlation coefficient
    # NOTE: proportion burned wld be initial bare spots minus final bare spots
    fig2, ax2 = plt.subplots(1,1,figsize=(12,8))

    for i,simulation in enumerate(results):
        p_spreads = np.array(list(simulation.keys()),dtype=float)
        # find the number of iterations for each simulation at its' probability of spread
        niters = [len(value[0]) for key,value in simulation.items()] # assume all are same length

        # calculate the proportion of the population dead at end
        prop_dead = [value[0][-1] for key,value in simulation.items()]
        # calculate proportion immune at end
        prop_immune = [value[1][-1] for key,value in simulation.items()]
        prop_alive = [value[3][-1] for key,value in simulation.items()]
        survival_rate = [sum(nums) for nums in zip(prop_immune,prop_alive)]
        # get the correlation coefficient for each simulation
        #ccf = np.corrcoef(prop_burned,p_spreads)[0,1]
    
        # now plot on line plot
        ax1.plot(p_spreads,niters,'-o',label=f'Run {i}')
        # plot on scatter plot with correlation coefficients
        ax2.plot(p_spreads,prop_alive,'-o',label=f'Never Caught Virus (No Immunity)')
        ax2.plot(p_spreads,prop_immune,'-o',label=f'Immune')
        ax2.plot(p_spreads,prop_dead,'-o',label=f'Dead')
        ax2.plot(p_spreads,survival_rate,'-o',label=f'Total Surviving')

    # more plot formatting
    ax1.legend()
    ax2.legend()

    if var == "dead":
        # label the axes
        ax1.set_ylabel('Number of Iterations')
        ax1.set_xlabel('Probability of Death (Mortality Rate)')
        # label axes 
        ax2.set_xlabel('Probability of Death (Mortality Rate)')
        ax2.set_ylabel('Proportion of Forest Burned')

        ax1.set_title('Time for Disease to Stop Spreading vs Mortality Rate')
        ax2.set_title('Proportion of the Population Surviving vs Mortality Rate')
    if var == "vax":
        # label the axes
        ax1.set_ylabel('Number of Iterations')
        ax1.set_xlabel('Proportion of Population')
        # label axes 
        ax2.set_xlabel('Vaccination Rate')
        ax2.set_ylabel('Proportion of Population')

        ax1.set_title('Time for Disease to Stop Spreading vs Early Vaccination Rate')
        ax2.set_title('Final Immunity in Population vs Early Vaccination Rate')

    # make directory to save plots if it doesn't already exist
    Path(f'{OUTPATH}/{run_name}').mkdir(parents=True, exist_ok=True)
    fig1.savefig(f'{OUTPATH}/{run_name}/figure1_{var}.png')
    fig2.savefig(f'{OUTPATH}/{run_name}/figure2_{var}.png')

def question_one(nx,ny,prob_spread=1.0,prob_die=1.0):
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
    prob_spread : float
        probability of fire spreading
        defaults to 1.0 for this experiment because always spreads
    prob_die : float
        probability of a burning cell "dying" (named due to universality)
        defaults to 1.0 for this experiment because the tree always burns
        to the ground

    Return
    ------
    current_state : np.ndarray
        final state of the forest after all iterations
    """
    # set path to save file
    run_name = f'q1_{nx}x{ny}_forest' # directory name to save figures
    # initialise forest
    # set arbitrary start probability
    start = (nx//2,ny//2)
    current_state = initialise_simulation(nx,ny,start)
    
    niters = 0
    visualise_current_state(current_state,f'Iteration: {niters}',run_name,f'unit_test_step-{niters:.1f}')
    # keep looping til no cells can are spreaders
    while 3 in current_state:
        niters += 1  # keep count
        # spread!!!
        current_state = spread(current_state,prob_spread,prob_die,dead=1)
        visualise_current_state(current_state,f'Iteration: {niters}',run_name,f'unit_test_step-{niters:.1f}')

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
    results_dict : dict(np.array)
        dictionary containing the proportion bare, burning and forested at each iteration

    Figures
    -------    
    Generates plots of the initial and final forest state, and plots designed to explain 
    the behavior of the forest.
    """
    # Set globals for this run
    p_variable = np.arange(0.0,1.1,0.1) # array of probabilities to iterate over
    prob_start = 0.04 # start fire based on pseudo-random seeding probabilities
    
    run_name = f'q2_{nx}x{ny}_forest' # directory name to save figures
    # make directory to save plots if it doesn't already exist
    Path(f'{OUTPATH}/{run_name}').mkdir(parents=True, exist_ok=True)
    
    # initialise empty dictionaries for forested, bare and burned
    forested_dict, bare_dict, burning_dict = {}, {}, {}

    # catch exceptions to the prob to vary function
    if prob_to_vary not in {'spread','bare'}:
        raise(ValueError(f'prob_to_vary: {prob_to_vary} is invalid. Choose from "spread" or "bare"'))

    # pass in the fixed and variable probability
    for prob in p_variable:
        if prob_to_vary == 'spread':    
            current_state = initialise_simulation(nx,ny,prob_start,fixed_prob)
        else:
            current_state = initialise_simulation(nx,ny,prob_start,prob)
            
        # create lists to save output for results dictionary
        forested_list, bare_list, burning_list = [],[],[]

        while 3 in current_state:
            if prob_to_vary == 'spread':
                current_state = spread(current_state,prob,1,dead=1) # prob that a cell burns to the ground is always 1
            else:
                current_state = spread(current_state,fixed_prob,1,dead=1) # prob that a cell burns to the ground is always 1

            # calculate and save proportion of the cells in each state for plotting
            forested_list.append((np.size(current_state[current_state == HEALTHY])/
                                                        (np.size(current_state))))
                    
            bare_list.append((np.size(current_state[current_state == IMMUNE])/
                                                    (np.size(current_state))))
                
                    
            burning_list.append((np.size(current_state[current_state == SPREADING])/
                                                        (np.size(current_state))))
                    
        forested_dict[f'{prob:.1f}'] = forested_list # key indicates probability of spread
        bare_dict[f'{prob:.1f}'] = bare_list # key indicates probability of spread
        burning_dict[f'{prob:.1f}'] = burning_list # key indicates probability of spread
    
    #----------#    
    # PLOTTING #
    #----------#
    # FIGURE 1: Probability the fire will spread vs the number of iterations
    # for the fire to burn out fully
    fig1, ax1 = plt.subplots(1,1,figsize=(12,8))  # set up figure and axes
    
    p_spreads = np.array(list(forested_dict.keys()),dtype=float)
    niters = [len(value) for key,value in forested_dict.items()] # assume all are same length

    ax1.plot(p_spreads,niters,'-o')

    # label the axes
    ax1.set_ylabel('Number of Iterations')
    ax1.set_xlabel('Probability that Fire Spreads')
    ax1.set_title('Iterations for Fire to Burn Out vs Probability of Fire Spreading')

    fig1.savefig(f'{OUTPATH}/{run_name}/prob_spread_vs_niters.png')

    # FIGURE 2: Proportion of the forest burned at the end vs the probability that
    # the fire will spread, plus linear correlation coefficient
    fig2, ax2 = plt.subplots(1,1,figsize=(12,8))

    # calculate the proportion of the forest burned at the end (final bare - inital bare)
    prop_burned = [(value[-1]-value[0]) for key,value in bare_dict.items()]
    # get the correlation coefficient for each simulation
    ccf = np.corrcoef(prop_burned,p_spreads)[0,1]
    ax2.plot(p_spreads,prop_burned,'-o',label=f'$R^2$ = {ccf:.2f}')

    ax2.set_xlabel('Probability that Fire Spreads')
    ax2.set_ylabel('Proportion of Forest Burned')
    ax2.set_title('Proportion of the Forest Burned vs Probability of a Given Cell Starting Bare')

    fig2.savefig(f'{OUTPATH}/{run_name}/prob_spread_vs_prop_burned.png')

    # FIGURE 3: heatmaps for each timestep and probability of spread
    # NOTE: need to improve formatting and 
    fig3, ax3 = plt.subplots(3,1,figsize=(10,12),sharey=True)
    # construct heatmap shapes
    hmap_length = np.max(niters)  # longest number of iterations wide
    hmap_width = len(p_spreads) # one row for each probability of spread

    # initialise 2D array
    hmap_arr = np.zeros((3,hmap_length,hmap_width),dtype=float)
    # populate hmap_arr with probabilities
    for row,(forest,bare,burn) in enumerate(zip(forested_dict.values(),
                                                bare_dict.values(),burning_dict.values())):
        
        hmap_arr[0,:len(forest),row] = forest # fill the forest part of the array
        hmap_arr[1,:len(bare),row] = bare # fill the bare part of the array
        hmap_arr[2,:len(burn),row] = burn # fill the burning part of the array

    # heatmap of forested cells throughout simulation    
    forested_contour = ax3[0].imshow(hmap_arr[0,:,:].T,cmap='Greens',vmax=1.0,vmin=0.0)
    plt.colorbar(forested_contour,ax=ax3[0],orientation='vertical',shrink=0.5,aspect=5)
    ax3[0].set_title('Proportion Forested',loc='left')
    ax3[0].set_yticks(np.arange(0,11,1),labels=p_spreads) # only need to do once because sharey=True

    # heatmap of bare cells throughout simulation
    bare_contour = ax3[1].imshow(hmap_arr[1,:,:].T,cmap='Greys',vmax=1.0,vmin=0.0)
    plt.colorbar(bare_contour,ax=ax3[1],orientation='vertical',shrink=0.5,aspect=5)
    ax3[1].set_title('Proportion Bare',loc='left')

    # heatmap of bare cells throughout simulation
    forested_contour = ax3[2].imshow(hmap_arr[2,:,:].T,cmap='Reds',vmax=1.0,vmin=0.0)
    plt.colorbar(forested_contour,ax=ax3[2],orientation='vertical',shrink=0.5,aspect=5)
    ax3[2].set_title('Proportion Burning',loc='left')

    fig3.tight_layout()

    fig3.savefig(f'{OUTPATH}/{run_name}/prob_spread_forest_heatmap.png')

    # populate the heatmap array
    return forested_dict

    
    

def question_three(nx,ny,fixed_prob=0.0,p_variable=np.arange(0.0,1.1,0.1),prob_to_vary='dead',ens=1,vis=False):
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
    ens : int
        number of ensemble members to generate
        defaults to 1 (e.g. deterministic run)

    Returns
    -------
    results_dict : dict(np.array)
        dictionary containing the proportion bare, burning and forested at each iteration

    Figures
    -------    
    Generates plots of the initial and final forest state, and plots designed to explain 
    the behavior of the forest.
    """
    
    prob_start = 0.03 # start fire

    run_name = f'q2_{nx}x{ny}_disease_2' # directory name to save figures
    
    # store output of simulations
    sims = []
    # catch exceptions to the prob to vary function
    if prob_to_vary not in {'dead','vax'}:
        raise(ValueError(f'prob_to_vary: {prob_to_vary} is invalid. Choose from "dead" or "vax"'))

    for sim in range(ens):
        # store proportion of grid in each state at each iteration
        # less memory intensive than constantly saving the full array
        results_dict = {}
        # pass in the fixed and variable probability
        for prob in p_variable:
            if prob_to_vary == 'dead':    
                current_state = initialise_simulation(nx,ny,prob_start,fixed_prob)
                visualise_current_state(current_state,f'death rate = {prob:.1f}',run_name,f'init_dead_{prob:.1f}')
            else:
                current_state = initialise_simulation(nx,ny,prob_start,prob)
                visualise_current_state(current_state,f'vaccine rate = {prob:.1f}',run_name,f'init_vax_{prob:.1f}')
            
            # create lists to save output for results dictionary
            dead, immune, healthy, spreading = [],[],[],[]
            niters = 1
            while 3 in current_state:
                if prob_to_vary == 'vax':
                    current_state = spread(matrix=current_state,p_spread=.8,p_death=0.0) # prob that a person always dies is 0%
                else:
                    current_state = spread(matrix=current_state,p_spread=.8,p_death=prob) # prob that a cell dies varies

                if vis:
                    visualise_current_state(current_state,f'step = {niters}',run_name,f'{prob_to_vary}-{prob:.1f}_step{niters:.1f}')

                # calculate and save proportion of the cells in each state for plotting
                dead.append((np.size(current_state[current_state == DEAD])/
                                                        (np.size(current_state))))
                    
                immune.append((np.size(current_state[current_state == IMMUNE])/
                                                        (np.size(current_state))))
                    
                spreading.append((np.size(current_state[current_state == SPREADING])/
                                                    (np.size(current_state))))
                
                    
                healthy.append((np.size(current_state[current_state == HEALTHY])/
                                                        (np.size(current_state))))
                
                niters +=1
                    
            results_dict[f'{prob:.1f}'] = (dead,immune,spreading,healthy) # key indicates probability of spread
            visualise_current_state(current_state,f'{prob_to_vary} rate = {prob:.1f}',run_name,f'final{prob_to_vary}_{prob:.1f}')
        
        sims.append(results_dict)
    # make the desired plots for question 2
    q3_plots(sims,run_name,prob_to_vary)
            
    return sims

# run simulations for question 1
question_one(3,3)
question_one(3,5)

# run simulations for question 2
f_dict = question_two(250,250)
#question_two(250,250,fixed_prob=1.0,prob_to_vary='bare')

# run simulations for question 3
#simulation_5 = question_three(250,250,fixed_prob=0,ens=1,prob_to_vary='vax',vis=False)
#simulation_6 = question_three(250,250,fixed_prob=0,ens=1,prob_to_vary='dead')