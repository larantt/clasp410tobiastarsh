#!/usr/bin/env python3
"""
lab01.py

Author : Lara Tobias-Tarsh
Last Modified : 9/11/24

The lab demonstrates how the principle of universality can be used to simulate a number of scenarios, 
in this case the spread of forest fires or diseases.

File contains the functions to execute a number of test simulations to prove this. To execute
this file, simply run:

```
python3 lab01.py
```
"""
#############
## IMPORTS ##
#############
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path

plt.ion()

#############
## GLOBALS ##
#############
outpath = '/Users/laratobias-tarsh/Documents/fa24/clasp410tobiastarsh/labs/lab01/figures'
# cell values
dead = 0
immune = 1
healthy = 2
spreading = 3
# Generate our custom segmented color map for this project.
forest_cmap = ListedColormap(['tan', 'lightsteelblue', 'darkgreen', 'crimson'])

###############
## FUNCTIONS ##
###############
def decide_fate(p_die,immune=immune,dead=dead):
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

def spread(matrix, p_spread, p_death, avail=healthy, spreader=spreading, immune=immune, dead=dead):
    """
    function finds the indices of all adjacent cells to the burning/infectious cell.

    Uses concept of universality again through the decide_fate() function which can
    be used to either kill/immunise a cell that is spreading, or infect/not infect
    a neighbouring cell.

    Parameters
    ----------
    matrix : np.ndarray
     multidimensional array representing the model grid
    avail : int
     integer representing a cell that could burn/be infected
    spreader :  int
     integer representing a cell that can spread fire/disease

    Returns
    -------
    spread_locs = list(tuple)
     list of tuples containing indicies of possible spread locations

    Example Usage
    -------------
    >
    >
     
    """
    next_matrix = matrix.copy() # copy the current state so this can be edited
    
    # get the size of the matrix to check boundary conditions
    m, n= np.array(matrix.shape)
    
    # loop over cells which are "spreaders"
    for i,j in np.argwhere(matrix==spreader):
        # if the position is not on the border, and it is a tree
        if i > 0:
            if matrix[i-1,j] == avail:
                # assign copy spreader value
                next_matrix[i-1,j] = decide_fate(p_die=p_spread,immune=healthy,dead=spreader)
        
        # if the position is not on the border, and it is a tree
        if i+1 < m:
            if matrix[i+1,j] == avail:
                # assign copy spreader value
                next_matrix[i+1,j] = decide_fate(p_die=p_spread,immune=healthy,dead=spreader)
        
        # if the position is not on the border, and it is a tree
        if j > 0:
            if matrix[i,j-1] == avail:
                # assign copy spreader value
                next_matrix[i,j-1] = decide_fate(p_die=p_spread,immune=healthy,dead=spreader)
        
        # if the position is not on the border, and it is a tree
        if j+1 < n:
            if matrix[i,j+1] == avail:
                # assign copy spreader value
                next_matrix[i,j+1] = decide_fate(p_die=p_spread,immune=healthy,dead=spreader)
    
        # finally, decide fate of the spreader cell
        next_matrix[i,j] = decide_fate(p_death,immune=immune,dead=dead)

    # return new state of fire
    return next_matrix

def initialise_simulation(ni,nj,p_start,p_bare,bare=immune,spreader=spreading,healthy=healthy,testing=False):
    """
    Function initialises forest fire/disease spread simulation based on a set of 
    probabilities for the initial state of the grid

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

    kwargs**
    bare : int
        value indicating a cell is bare
    spreader : int
        value indicating a cell can spread fire/disease
    healthy : int
        value indicating a cell is healthy
    testing : bool
        turns off random start and bare probabilities
        if True, all cells are healthy and center cell is on fire

    Returns
    -------
    initial_matrix : np.ndarray
        2D array with initial state of the simulation

    """
    # generate healthy forest/population
    initial_matrix = np.zeros((ni,nj),dtype=int) + healthy
    
    # start some fires
    if testing:
        # index into the center of the array and start fire
        # code from https://stackoverflow.com/questions/15257277/index-the-middle-of-a-numpy-array
        # NOTE: for a non-symmetrical array the integer division will place CLOSE to center but there is no real central index
        np.put(initial_matrix, initial_matrix.size // 2, 3)
    else:
        # create bare spots/immunity
        isbare = np.random.rand(ni,nj) < p_bare
        initial_matrix[isbare] = bare
        # start some random fires/infect some randos
        start = np.random.rand(ni,nj) < p_start
        initial_matrix[start] = spreader
    
    # return initial state of the simulation
    return initial_matrix

def visualise_current_state(matrix,step,run_name):
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
    fig, ax = plt.subplots(1, 1) # set up the figure
    contour = ax.matshow(matrix, vmin=0, vmax=3,cmap=forest_cmap) # plot the forest
    ax.set_title(f'Iteration = {step:03d}',loc='left') # set title
    plt.colorbar(contour, ax=ax)

    # make directory to save plots if it doesn't already exist
    Path(f'{outpath}/{run_name}').mkdir(parents=True, exist_ok=True)
    # save the figure at each iteration
    fig.savefig(f'{outpath}/{run_name}/step_{step}.png')

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
    run_name = f'test_{nx}x{ny}_forest' # directory name to save figures
    # initialise forest
    # set arbitrary start and bare probabilities, as testing=True will override them
    current_state = initialise_simulation(nx,ny,1.0,1.0,testing=True)
    
    niters = 0
    visualise_current_state(current_state,niters,run_name)
    # keep looping til no cells can are spreaders
    while 3 in current_state:
        niters += 1  # keep count
        # spread!!!
        current_state = spread(current_state,prob_spread,prob_die)
        visualise_current_state(current_state,niters,run_name)
        
        
    return current_state


simulation_1 = question_one(3,3)
simulation_2 = question_one(3,4)