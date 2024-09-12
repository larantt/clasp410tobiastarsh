#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path

plt.ion()

#############
## GLOBALS ##
#############
nx,ny = 10,7 
prob_spread = 0.7 # Chance to spread to adjacent cells.
prob_bare = 0.0   # Chance of cell to start as bare patch.
prob_start = 0.1  # Chance of cell to start on fire.
prob_die = 0.5    # Probability a cell dies
TEST = False      # should the simulation be run in test mode.
outpath = '/Users/laratobias-tarsh/Documents/fa24/clasp410tobiastarsh/labs/lab01/figures'

# user inputs
run_name = f'test_{nx}x{ny}_disease' # directory name to save figures

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
def decide_fate(p_die=prob_die,immune=immune,dead=dead):
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

def spread(matrix, p_spread=prob_spread, p_death=prob_die, avail=healthy, spreader=spreading, immune=immune, dead=dead):
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

def visualise_current_state(matrix,step):
    fig, ax = plt.subplots(1, 1)
    contour = ax.matshow(matrix, vmin=0, vmax=3,cmap=forest_cmap)
    ax.set_title(f'Iteration = {step:03d}')
    plt.colorbar(contour, ax=ax)

    Path(f'{outpath}/{run_name}').mkdir(parents=True, exist_ok=True)
    fig.savefig(f'{outpath}/{run_name}/step_{step}.png')

def main():
    # initialise forest
    current_state = initialise_simulation(nx,ny,prob_start,prob_bare,testing=TEST)
    
    niters = 0
    visualise_current_state(current_state,niters)
    # keep looping til no cells can are spreaders
    while 3 in current_state:
        niters += 1  # keep count
        # spread!!!
        current_state = spread(current_state)
        visualise_current_state(current_state,niters)
        
        
    return current_state


def question_one(nx,ny,prob_spread=1.0,prob_bare=0,prob_die=1.0):
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
    
    """

current_state = main()