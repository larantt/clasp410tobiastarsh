#!/usr/bin/env python3
"""
lab02.py

This lab investigates the stability of a high and low order ODE solver 
(the Euler Method and the DOP853 RK8 solver from Scipy) when solvinga a
set of coupled equations (the Lotka-Volterra equations) for competition
and predator-prey relationships.

This lab produces the following figures:
q1_competition.png  : plot showing how timestep affects the stability of solvers
                      for the competition equations

q1_pred_prey.png    : plot showing how timestep affects the stability of solvers
                      for the predator-prey equations

q2_panel.png        : panel plot showing the impact of varying initial conditions
                      and coefficients of the competition equations impacts the 
                      behavior and final population of each species

q2_final_coeffs.png : panel plot showing how varying the coefficients can affect
                      the final population of a given set of competition equations

q3_panel.png        : panel plot showing the impact of varying initial conditions
                      and coefficients of the predator-prey equations impacts the 
                      behavior and final population of each species

q3_phase.png        : phase plot for each set of initial conditions in q3_panel.png

unit_test.png       : unit test which reproduces the figure in the lab manual


USER INPUTS
-----------
OUTPATH : str
    absolute filepath to the directory where figures should be saved.
    this should be edited for the machine you are working on.

To run this file, execute the following command:
>>> python3 lab02.py

Ensure you have changed the OUTPATH global to your machine.

This should produce all figures in the lab report and save them to disk.

NOTE: when solving for the second set of coefficients in question 3, a runtime warning occurs:
  ```  
    RuntimeWarning: overflow encountered in scalar multiply dN1_dt = a*N[0] - b*N[0]*N[1]
    RuntimeWarning: overflow encountered in scalar multiply dN2_dt = -1*c*N[1] + d*N[0]*N[1]
    RuntimeWarning: invalid value encountered in scalar add dN2_dt = -1*c*N[1] + d*N[0]*N[1]
   ```
This is due to numerical instability in the Euler method. However, the Euler method solutions are
NOT plotted, or relevant to reproduce the contents of the lab. As a result, these were not addressed
as this is expected behavior and alerts the user to the inherent instability of the Euler method as a 
first order solver. The rk8 solver could be called alone here, but because the data structure to store output
is created in the solve_eqns function, which uses both methods, it was more convenient not to.
"""
###########
# IMPORTS #
###########
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

# Changing font to stix; setting specialized math font properties as directly as possible
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.it'] = 'STIXGeneral:italic'
plt.rcParams['mathtext.bf'] = 'STIXGeneral:italic:bold'

plt.ion()  # comment out when not testing

# set the few globals to be used in this lab
# define location to save figures - change on your machine as necessary
OUTPATH = '/Users/laratobias-tarsh/Documents/fa24/clasp410tobiastarsh/labs/lab02/figures'
# the best colorblind friendly colormap (https://davidmathlogic.com/colorblind/)
IBM_cmap = ["#648FFF","#DC267F","#FFB000"] 

#########################
# INTEGRATION FUNCTIONS #
#########################

def int_to_roman(number):
    """
    Helper function to convert integers to roman numerals. Used for nice figure labelling with
    multipanel plots.

    Heavily inspired by https://stackoverflow.com/questions/28777219/basic-program-to-convert-integer-to-roman-numerals

    Parameters
    ----------
    number : int
        number to convert to roman numerals
    
    Returns
    -------
        : str
        string of the roman numeral for that plot
    """
    numerals = [
        (1000, "M"),
        ( 900, "CM"),
        ( 500, "D"),
        ( 400, "CD"),
        ( 100, "C"),
        (  90, "XC"),
        (  50, "L"),
        (  40, "XL"),
        (  10, "X"),
        (   9, "IX"),
        (   5, "V"),
        (   4, "IV"),
        (   1, "I"),
    ]
    result = []
    for (arabic, roman) in numerals:
        (factor, number) = divmod(number, arabic)
        result.append(roman * factor)
        if number == 0:
            break
    return "".join(result)

def lv_comp(t, N, a=1, b=2, c=1, d=3):
    '''
    This function calculates the Lotka-Volterra competition equations for
    two species. Given normalized populations, `N1` and `N2`, as well as the
    four coefficients representing population growth and decline,
    calculate the time derivatives dN_1/dt and dN_2/dt and return to the
    caller.

    This function accepts `t`, or time, as an input parameter to be
    compliant with Scipy's ODE solver. However, it is not used in this
    function.

    Parameters
    ----------
    t : float
        The current time (not used here).
    N : two-element list
        The current value of N1 and N2 as a list (e.g., [N1, N2]).
    a, b, c, d : float, defaults=1, 2, 1, 3
        The value of the Lotka-Volterra coefficients.
    Returns
    -------
    dN1_dt, dN2_dt : floats
        The time derivatives of `N1` and `N2`.
    '''
    # Here, N is a two-element list such that N1=N[0] and N2=N[1]
    dN1_dt = a*N[0]*(1-N[0]) - b*N[0]*N[1]
    dN2_dt = c*N[1]*(1-N[1]) - d*N[1]*N[0]
    return dN1_dt, dN2_dt

def lv_pred_prey(t, N, a=1, b=2, c=1, d=3):
    '''
    This function calculates the Lotka-Volterra predator-prey equations for
    two species. Given normalized populations, `N1` and `N2`, as well as the
    four coefficients representing population growth and decline,
    calculate the time derivatives dN_1/dt and dN_2/dt and return to the
    caller.

    This function accepts `t`, or time, as an input parameter to be
    compliant with Scipy's ODE solver. However, it is not used in this
    function.

    Parameters
    ----------
    t : float
        The current time (not used here).
    N : two-element list
        The current value of N1 and N2 as a list (e.g., [N1, N2]).
    a, b, c, d : float, defaults=1, 2, 1, 3
        The value of the Lotka-Volterra coefficients.
    Returns
    -------
    dN1_dt, dN2_dt : floats
        The time derivatives of `N1` and `N2`.
    '''
    # Here, N is a two-element list such that N1=N[0] and N2=N[1]
    dN1_dt = a*N[0] - b*N[0]*N[1]
    dN2_dt = -1*c*N[1] + d*N[0]*N[1]
    return dN1_dt, dN2_dt

def euler_solve(func, N1_init=.5, N2_init=.5, dT=.1, t_final=100.0,**kwargs):
    '''
    Given a function representing the first derivative of f, an
    initial condiition f0 and a timestep, dt, solve for f using
    Euler's method.

    Parameters
    ----------
    func : function
        system of equations to be input into euler solver
    N1_init : float
        start value for N1 differential equation
    N2_init : float
        start value for N2 differential equation
    dT : float
        timestep to run the Euler solver at
    t_final : float
        timestep to terminate the differentiation at
        (equivalent of max iters)

    kwargs
    ------
    a, b, c, d : float, defaults=1, 2, 1, 3
        The value of the Lotka-Volterra coefficients.
        Can be passed to change initial conditions.

    Returns
    -------
    t : arrayLike
        array of timesteps
    f_N1 : arrayLike
        array of values for the N1 differential equation
    f_N2 : arrayLike
        array of values for the N2 differential equation
    '''

    # initialise empty arrays for the time and differetial equations
    t = np.arange(0.0, t_final, dT)
    f_N1, f_N2 = np.zeros(t.size),np.zeros(t.size)
    # set the initial state of the Euler solver
    f_N1[0], f_N2[0] = N1_init, N2_init

    # integrate the function
    for i in range(t.size-1):
        # get the derivatives of the function
        dN1dt, dN2dt = func(i,[f_N1[i],f_N2[i]],**kwargs) # passing kwargs to change initial conditions
        # integrate the first differential equation
        f_N1[i+1] = f_N1[i] + dT * dN1dt
        # integrate the second differential equation
        f_N2[i+1] = f_N2[i] + dT * dN2dt

    # Return values to caller
    return t, f_N1, f_N2

def solve_rk8(func, N1_init=.5, N2_init=.5, dT=10, t_final=100.0, a=1, b=2, c=1, d=3):
    '''
    Solve the Lotka-Volterra competition and predator/prey equations using
    Scipy's ODE class and the adaptive step 8th order solver.

    Parameters
    ----------
    func : function
    A python function that takes `time`, [`N1`, `N2`] as inputs and
        returns the time derivative of N1 and N2.
    N1_init, N2_init : float
        Initial conditions for `N1` and `N2`, ranging from (0,1]
    dT : float, default=10
        Largest timestep allowed in years.
    t_final : float, default=100
        Integrate until this value is reached, in years.
    a, b, c, d : float, default=1, 2, 1, 3
        Lotka-Volterra coefficient values

    Returns
    -------
    time : Numpy array
        Time elapsed in years.
    N1, N2 : Numpy arrays
        Normalized population density solutions.
    [a,b,c,d] : list
        list of the parameters used in the equation
    '''
    # Configure the initial value problem solver
    result = solve_ivp(func, [0, t_final], [N1_init, N2_init],
                        args=[a, b, c, d], method='DOP853', max_step=dT)
    # Perform the integration
    time, N1, N2 = result.t, result.y[0, :], result.y[1, :]
    
    # save as a dictionary for ease
    params = {
        'a'  : a,
        'b'  : b,
        'c'  : c,
        'd'  : d,
        'dT' : dT,
        'Initial Conditions' : f'N1 = {N1_init:.2f}, N2 = {N2_init:.2f}'
    }

    # Return values to caller.
    return time, N1, N2, params

def line_plot(ax,data_dict,lws=3,format=True,euler=True):
    """
    Function creates phase plots to show the output of
    each ODE solver when solving the Lotka-Volterra equations.

    Function is intended to do all of the plotting and 
    formatting on a supplied axis, therefore note that the
    figure and axes need to be created outside of this helper
    function. This is intended to allow for maximum flexibility
    when making figures while reducing additional code.

    Parameters
    ----------
    ax : mpl.Axes
        axis to create the phase plot on
    data_dict : dict
        dictionary containing all the information for the plot
    lws : int
        allows linewidth to be changed
    format : bool
        Should the axes be formatted with labels and a legend
        Defaults to True (formatting on)
    euler : bool
        Should the Euler equations be plotted
        Defaults to True (Euler equations plotted)

    Returns
    -------
    lines : list(mpl.Line2D)
        list of lines on the line plot to be used for formatting
    """
    # NOTE: we know the keys because the dictionary follows a standard specified format

    # if we want to plot rk8 and the euler method answers
    if euler:
        # plot euler equation for N1
        n1_e, = ax.plot(data_dict['times_euler'],data_dict['N1_euler'],
                lw=lws,c='cornflowerblue',label=r'$\mathregular{N_1}$ Euler')
        # plot euler equation for N2
        n2_e, = ax.plot(data_dict['times_euler'],data_dict['N2_euler'],
                lw=lws,c='indianred',label=r'$\mathregular{N_2}$ Euler')
        # plot rk8 equation for n1
        n1_r, = ax.plot(data_dict['times_rk8'],data_dict['N1_rk8'],
            lw=lws,c='cornflowerblue',label=r'$\mathregular{N_1}$ RK8',ls='--')
        # plot rk8 equation for n2
        n2_r, = ax.plot(data_dict['times_rk8'],data_dict['N2_rk8'],
            lw=lws,c='indianred',label=r'$\mathregular{N_2}$ RK8',ls='--')
        # return the lines for legend formatting
        lines = [n1_e,n2_e,n1_r,n2_r]
    
    # if we dont want the euler method answers, just rk8
    else:
        # plot n1
        n1_r, = ax.plot(data_dict['times_rk8'],data_dict['N1_rk8'],
            lw=lws,c='cornflowerblue',label=r'$\mathregular{N_1}$ RK8')
        # plot n2
        n2_r, = ax.plot(data_dict['times_rk8'],data_dict['N2_rk8'],
            lw=lws,c='indianred',label=r'$\mathregular{N_2}$ RK8')
        # return lines for legend formatting
        lines = [n1_r,n2_r]

    # if we want formatting for both axes
    if format:
        # set legend and format nicely
        legend_props = {"size" : 12, "weight" : 'bold'}
        ax.legend(prop=legend_props,loc='upper left')
        
        # set labels and title and format nicely
        labkw = dict(size=14,weight='bold') # dictionary formats the axes labels to look nice
        ax.set_xlabel(r'Time $\mathbf{(years)}$', **labkw)
        ax.set_ylabel('Population/Carrying Capacity', **labkw)
        ax.set_title(f"Lotka-Volterra {data_dict['model']} Model", fontsize=18, fontweight='bold')

        # do extra formatting manually because i have a crippling need to be in control of everything at all times :)
        for xlab,ylab in zip(ax.get_xticklabels(),ax.get_yticklabels()):
            xlab.set_weight('bold')
            ylab.set_weight('bold')
        
    # return the axes and legend handles
    return lines

def solve_eqns(model='Competition',**kwargs):
    """
    Function solves Lotka-Volterra equations using both the euler and rk8 solvers
    and a set of defined parameters which are either taken from the defaults or
    passed in as kwargs. It then formats the data into a dictionary which is 
    used throughout the lab to store data.

    Parameters
    ----------
    model : string
        either 'Competition' or 'Predator-Prey'
        defaults to 'Competition'
    
    **kwargs
    ---------
    # note to format as a dict, look up how to document appropriately
    a, b, c, d : float, defaults=1, 2, 1, 3
        The value of the Lotka-Volterra coefficients.
    N1_init : float
        start value for N1 differential equation
    N2_init : float
        start value for N2 differential equation
    dT : float
        timestep to run the Euler solver at
    t_final : float
        timestep to terminate the differentiation at
        (equivalent of max iters)

    Returns
    -------
    data_dict : dict
        Dictionary containing output for both datasets
    """
    if model == 'Competition':
        # solve competition equations
        t_euler,n1_euler,n2_euler = euler_solve(lv_comp,**kwargs)
        t_rk8, n1_rk8, n2_rk8, params = solve_rk8(lv_comp,**kwargs)
    elif model == 'Predator-Prey':
        # solve predator-prey equations
        t_euler,n1_euler,n2_euler = euler_solve(lv_pred_prey,**kwargs)
        t_rk8, n1_rk8, n2_rk8, params = solve_rk8(lv_pred_prey,**kwargs)
    else:
        raise(ValueError(f'model={model} of type{type(model)} is incorrect.\nSelect between str(Competition) or str(Predator-Prey)'))
    
    # get the longest time array (note that the scipy solver might be longer/shorter bc of adaptive timestepping)
    # NOTE: this feels like unsafe coding, should look for a better way later
    #times = t_rk8 if max(len(t_rk8),len(t_euler)) == len(t_rk8) else t_euler
    # now format into dictionary
    data_dict = {
        'model'      : model,
        'params'     : params,
        'times_euler': t_euler,
        'N1_euler'   : n1_euler,
        'N2_euler'   : n2_euler,
        'times_rk8'  : t_rk8,   
        'N1_rk8'     : n1_rk8,
        'N2_rk8'     : n2_rk8
    }

    return data_dict

def unit_test():
    """
    This function performs a unit test for both sets of
    Lotka-Volterra equations, which matches the figure provided
    in the lab. It is also helpful for demonstrating the dictionary
    format in which data for this lab is stored.

    All parameters in the functions are explicitly passed for the
    purpose of demonstration, though note that calling each of the
    functions without input will produce the same results.

    In this unit test, the coefficients are as follows:
    a = 1, b = 2, c = 1, d = 3
    N1_init = 0.3, N2_init = 0.6
    t_final = 100
    
    For predator-prey: dT = 0.05
    For competition:   dT = 1.0

    Parameters
    ----------
    None

    Returns
    -------
    None

    """
    # define the kwargs for the equation set
    inputs = {
        'a'       : 1,
        'b'       : 2,
        'c'       : 1,
        'd'       : 3,
        'N1_init' : 0.3,
        'N2_init' : 0.6,
        't_final' : 100
    }
    # next solve the equations for each model
    competition_data = solve_eqns('Competition',dT=1.0,**inputs)
    pred_pray_data = solve_eqns('Predator-Prey',dT=0.05,**inputs)

    # now plot the data for confirmation of performance
    fig, ax = plt.subplots(1,2,figsize=(15,8))
    line_plot(ax[0],competition_data)
    line_plot(ax[1],pred_pray_data)

    # add parameters as text on figure
    coeff_text = [f"{key}={val}" for key,val in competition_data["params"].items()][0:-1] # skip dT
    # annotate figure with the coefficients neatly
    ax[0].text(100,-0.14,f'Coefficients: {" ".join(str(i) for i in coeff_text)}',
               fontsize=10,fontweight='bold')
    
    # make directory to save plots if it doesn't already exist
    Path(f'{OUTPATH}').mkdir(parents=True, exist_ok=True)
    # save the figure
    fig.savefig(f'{OUTPATH}/unit_test.png',dpi=300)

def question_one():
    """
    Contains all necessary code to answer question 1 in the lab.

    First, executes the unit test, then creates two four panel
    plots. The first explores the result of varying the timestep
    on the competition equations, the second explores the result
    of varying the timestep on the predator-prey equations.

    Each plot is a 3x2 panel showing select example initial conditions.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    # first execute the unit test to verify code is working
    unit_test()       # dont store output bc who cares

    # now, varying the timestep to generate ensembles
    inputs = {
        'a'       : 1,
        'b'       : 2,
        'c'       : 1,
        'd'       : 3,
        'N1_init' : 0.3,
        'N2_init' : 0.6,
        't_final' : 100
    }

    # plot the full ensemble of times
    dts_comp = np.array([0.01,0.1,0.25,1.5,1.75,1.95])
    dts_pred = np.array([0.01,0.02,0.03,0.05,0.07,0.09])
    competition_data = np.array([solve_eqns('Competition',dT=time,**inputs) for time in dts_comp])
    pred_prey_data = np.array([solve_eqns('Predator-Prey',dT=time,**inputs) for time in dts_pred])

    # now plot select timesteps individually for comparison
    # first do small dTs
    fig1, ax1 = plt.subplots(2,3,figsize=(18*2,12))

    for ax,data in zip(ax1.flatten(), competition_data):
        lines = line_plot(ax,data,lws=2,format=False)
        ax.set_title(f'Timestep: {data["params"]["dT"]} Years',fontweight='bold',fontsize=12)

    # add parameters as text on figure
    coeff_text = [f"{key}={val}" for key,val in competition_data[0]["params"].items()][0:-2] # skip dT, initial conditions
    # annotate figure with the coefficients neatly
    ax1[1,1].text(18,-0.25,f'Coefficients: {" ".join(str(i) for i in coeff_text)}',
               fontsize=10,fontweight='bold')
    
    # figure formatting
    fig1.legend(lines,[line.get_label() for line in lines],fontsize=12,loc='upper center',
                ncols=4,frameon=False,bbox_to_anchor=(0.5,0.972))
    
    # set overall title
    fig1.suptitle('Impact of Timestep on Numerical Solver for Lotka-Volterra Competition Equations',fontweight='bold',fontsize=18)

    labkw = dict(size=16,weight='bold') # dictionary formats the axes labels to look nice
    ax1[1,1].set_xlabel(r'Time $\mathbf{(years)}$', **labkw)       # set common x label
    ax1[0,0].set_ylabel('Population/Carrying Capacity',y=-0.3, **labkw)   # set common y label
    
    # make things fiit nicely
    fig1.tight_layout(h_pad=-2)

    # make directory to save plots if it doesn't already exist
    Path(f'{OUTPATH}').mkdir(parents=True, exist_ok=True)
    # save the figure
    fig1.savefig(f'{OUTPATH}/q1_competition.png',dpi=300)

    # now do the predator-prey equations
    fig2, ax2 = plt.subplots(2,3,figsize=(18*2,12))

    for ax,data in zip(ax2.flatten(), pred_prey_data):
        lines = line_plot(ax,data,lws=2,format=False)
        ax.set_title(f'Timestep: {data["params"]["dT"]} Years',fontweight='bold',fontsize=12)

    # add parameters as text on figure
    coeff_text = [f"{key}={val}" for key,val in pred_prey_data[0]["params"].items()][0:-2] # skip dT, initial conditions
    # annotate figure with the coefficients neatly
    ax2[1,1].text(18,-1.0,f'Coefficients: {" ".join(str(i) for i in coeff_text)}',
               fontsize=10,fontweight='bold')
    
    # figure formatting
    fig2.legend(lines,[line.get_label() for line in lines],fontsize=12,loc='upper center',
                ncols=4,frameon=False,bbox_to_anchor=(0.5,0.972))
    
    # set overall title
    fig2.suptitle('Impact of Timestep on Numerical Solver for Lotka-Volterra Predator-Prey Equations',fontweight='bold',fontsize=18)

    # set axes labels
    labkw = dict(size=16,weight='bold') # dictionary formats the axes labels to look nice
    ax2[1,1].set_xlabel(r'Time $\mathbf{(years)}$', **labkw)       # set common x label
    ax2[0,0].set_ylabel('Population/Carrying Capacity',y=-0.3, **labkw)   # set common y label

    fig2.tight_layout(h_pad=-2)
    # make directory to save plots if it doesn't already exist
    Path(f'{OUTPATH}').mkdir(parents=True, exist_ok=True)
    # save the figure
    fig2.savefig(f'{OUTPATH}/q1_pred_prey.png',dpi=300)

def solve_equilibria(a,b,c,d,model='Competition'):
    """
    Function solves equilibrium conditions for a given
    Lotka-Volterra equation, using its parameters.

    The formulas were derived by setting the derivatives to 0
    and rearranging for N1 and N2, as is demonstrated in the lab
    for the competition equations.

    Parameters
    ----------
    a, b, c, d : float
        Lotka-Volterra coefficient values
    model : str
        string stating which equilibria to solve for
        Defaults to 'Competition'

    Returns
    -------
    (N1, N2) : tuple
        tuple containing initial conditions that
        generate equilibrium for a given set of coefficients
    """
    if model == 'Competition':
        # solve competition equilibrium conditions
        N1 = (c*(a-b))/((c*a)-(b*d))
        N2 = (a*(c-d))/((c*a)-(b*d))
    elif model == 'Predator-Prey':
        # solve predator prey equilibrium conditions
        N1 = a/b
        N2 = c/d
    else:
        # throw error if inputs are invalid
        raise(ValueError(f'model={model} of type{type(model)} is incorrect.\nSelect between str(Competition) or str(Predator-Prey)'))
    return (N1,N2)

def question_two():
    """
    Contains the code to answer Question 2 in the lab

    First, creates a 4x3 panel plot of selected examples for different
    coefficients with different set initial conditions (including one solved equilibrium)
    in order to demonstrate how varying the coefficients and initial conditions 
    affect the behavior of the species.

    Next, creates a 2x2 panel plot showing the final results when each coefficient
    is varied to demonstrate the effect that varying coefficients has on the 
    final behavior

    Parameters
    ----------
    None

    Returns
    -------
    None

    """
    # part one - varying the initial conditions

    # define 3 sets of coeffs to w defined equilibria
    ex_a = {'a':2,'b':1,'c':3,'d':2}    # Stable equilibrium
    ex_b = {'a':2,'b':1,'c':2,'d':1}    # Both equal (stable)
    ex_c = {'a':2,'b':3,'c':1,'d':4}    # Unstable equilibrium

    # create a 4 part axis
    fig1, ax1 = plt.subplots(4,3,figsize=(13,10),sharex=True,sharey=True)

    # now loop over the axes rows and the sets of coefficients
    for col,coef in zip([0,1,2],[ex_a,ex_b,ex_c]):
         
        # define variable initial conditions for each scenario
        inits = [(0.2,0.7),(0.6,0.3),(0.5,0.5),solve_equilibria(**coef)]

        # now loop over the sets of axes and plot
        for row,init in zip([0,1,2,3],inits):
            # set formatting to false for more control, store labels
            data = solve_eqns('Competition',N1_init=init[0],N2_init=init[1],dT=0.5,**coef)
            lines = line_plot(ax1[row,col],data,lws=2,format=False)
            # set equal x limit
            ax1[row,col].set_ylim(-0.1,1.1)
            # label the figures with initial conditions
            ax1[row,col].set_title(f'N1={init[0]:.2f}, N2={init[1]:.2f}',fontsize=10,
                                   fontweight='bold',loc='left',y=0.98,c='#3b3b3b')

        # axes row formatting
        coeff_text = [f"{key}={val}" for key,val in coef.items()]
        ax1[0,col].set_title(f'Coefficients: {" ".join(str(i) for i in coeff_text)}',
                             fontsize=12,fontweight='bold',y=1.08)

    # loop over axes and add roman numeral for labelling
    for num,axis in enumerate(ax1.flatten()):
        # number the figure with roman numerals
            axis.set_title(int_to_roman(num+1),fontsize=10,
                                   fontweight='bold',loc='right',y=0.98,c='#3b3b3b')
    # figure formatting
    fig1.legend(lines,[line.get_label() for line in lines],fontsize=12,loc='upper center',
                ncols=4,frameon=False,bbox_to_anchor=(0.5,0.97))
    fig1.suptitle('Impact of Varying Initial Conditions on Lotka-Volterra Competition Equations',
                  fontsize=16,fontweight='bold')
    
    labkw = dict(size=16,weight='bold') # dictionary formats the axes labels to look nice
    ax1[3,1].set_xlabel(r'Time $\mathbf{(years)}$', **labkw)       # set common x label
    ax1[1,0].set_ylabel('Population/Carrying Capacity',y=-0.3, **labkw)   # set common y label
    
    # make things fiit nicely
    fig1.tight_layout(h_pad=-1)
    
    # make directory to save plots if it doesn't already exist
    Path(f'{OUTPATH}').mkdir(parents=True, exist_ok=True)
    # save the figure
    fig1.savefig(f'{OUTPATH}/q2_panel.png',dpi=300)

    # PART TWO - varying the coefficients

    # make axes
    fig2, ax2 = plt.subplots(2,2,figsize=(12,8),sharey=True)

    vary_coeffs = np.arange(0,4.1,0.1)
    # vary all the coeffs, list comps execute the equations
    vary_a = [solve_eqns('Competition',N1_init=0.5,N2_init=0.5,a=vary,b=1,c=2,d=1,dT=0.5) for vary in vary_coeffs]
    vary_b = [solve_eqns('Competition',N1_init=0.5,N2_init=0.5,a=2,b=vary,c=2,d=1,dT=0.5) for vary in vary_coeffs]
    vary_c = [solve_eqns('Competition',N1_init=0.5,N2_init=0.5,a=2,b=1,c=vary,d=1,dT=0.5) for vary in vary_coeffs]
    vary_d = [solve_eqns('Competition',N1_init=0.5,N2_init=0.5,a=2,b=1,c=2,d=vary,dT=0.5) for vary in vary_coeffs]

    # now loop over and plot the final populations for each one
    for ax,variable,lab in zip(ax2.flatten(),[vary_a,vary_b,vary_c,vary_d],['a','b','c','d']):
        # list comps store the final state of the data
        n1s = [data['N1_rk8'][-1] for data in variable]
        n2s = [data['N2_rk8'][-1] for data in variable]
        
        # now plot the final state for each initial condition
        n1, = ax.plot(vary_coeffs,n1s,c='cornflowerblue',label=r'$\mathregular{N_1}$ RK8')
        n2, = ax.plot(vary_coeffs,n2s,c='indianred',label=r'$\mathregular{N_2}$ RK8')
        
        # format the axes
        ax.set_title(f'Varying {lab}',fontsize=12,fontweight='bold',loc='left')
        ax.set_xlabel(f'Value of Coefficient {lab}',fontsize=12,fontweight='bold')

    # more axes formatting
    ax2[0,0].set_ylabel('Population/Carrying Capacity',fontsize=16,fontweight='bold',y=-0.3)

    # set legend
    fig2.legend([n1,n2],[n1.get_label(),n2.get_label()],loc='upper center',ncols=4,
                fontsize=12,frameon=False,bbox_to_anchor=(0.5,0.94))
    fig2.suptitle('Final Population Carrying Capacity Dependent on Coefficient\nBase State: a=2, b=1, c=2, d=1',
                  fontweight='bold',fontsize=16)
    
    # make things fit nicely
    fig2.tight_layout(h_pad=-1.5)
    # make directory to save plots if it doesn't already exist
    Path(f'{OUTPATH}').mkdir(parents=True, exist_ok=True)
    # save the figure
    fig2.savefig(f'{OUTPATH}/q2_final_coeffs.png',dpi=300)
  
def question_three():
    """
    Contains the code to answer Question 2 in the lab

    First, creates a 3x3 panel plot of selected examples for different
    coefficients with different set initial conditions (including one solved equilibrium)
    in order to demonstrate how varying the coefficients and initial conditions 
    affect the behavior of the species.

    Next, creates a 1x3 panel plot showing the phase diagrams for the inital conditions
    in the figure above

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    # define 3 sets of coeffs to w defined equilibria
    ex_a = {'a':1.,'b':1.,'c':1.,'d':1.}    # Stable equilibrium
    ex_b = {'a':2,'b':0.5,'c':3,'d':0.4}
    ex_c = {'a':0.5,'b':3.,'c':0.1,'d':2.}    # Both equal (stable)

    fig1, ax1 = plt.subplots(3,3,figsize=(13,10))
    fig1b, ax1b = plt.subplots(1,3,figsize=(15,4))
    # now loop over the axes rows and the sets of coefficients
    for col,coef in zip([0,1,2],[ex_a,ex_b,ex_c]):
         
        # define variable initial conditions for each scenario
        inits = [(0.2,0.7),(1,1),(5,2)]

        # now loop over the sets of axes and plot
        for row,init in zip([0,1,2],inits):
            # set formatting to false for more control, store labels
            data = solve_eqns('Predator-Prey',N1_init=init[0],N2_init=init[1],dT=0.05,**coef)
            lines = line_plot(ax1[row,col],data,lws=2,format=False,euler=False)
            
            ax1[row,col].set_title(f'N1={init[0]:.2f}, N2={init[1]:.2f}',fontsize=10,
                                   fontweight='bold',loc='left',y=0.98,c='#3b3b3b')
            
            
            # now plot the phase plots on their axes
            ax1b[col].plot(data["N1_rk8"],data['N2_rk8'],c=IBM_cmap[row],
                            label=f'N1={init[0]:.2f}, N2={init[1]:.2f}',lw=1)
        
            # just to aid visualisation
            if (coef == ex_a) & (init == (1,1)):
                ax1b[col].plot(data["N1_rk8"],data['N2_rk8'],'-x',c=IBM_cmap[row],
                                label=f'N1={init[0]:.2f}, N2={init[1]:.2f}',lw=1)

        # axes row formatting
        coeff_text = [f"{key}={val}" for key,val in coef.items()]
        ax1[0,col].set_title(f'Coefficients: {" ".join(str(i) for i in coeff_text)}',
                             fontsize=12,fontweight='bold',y=1.08)
        ax1b[col].set_title(f'Coefficients: {" ".join(str(i) for i in coeff_text)}',
                             fontsize=12,fontweight='bold',y=1.08)

    # loop over axes and add roman numeral for labelling
    for num,axis in enumerate(ax1.flatten()):
        # number the figure with roman numerals
            axis.set_title(int_to_roman(num+1),fontsize=10,
                                   fontweight='bold',loc='right',y=0.98,c='#3b3b3b')
            
    # loop over axes and add roman numeral for labelling
    for num,axis in enumerate(ax1b.flatten()):
        # number the figure with roman numerals
            axis.set_title(int_to_roman(num+1),fontsize=10,
                                   fontweight='bold',loc='right',y=0.98,c='#3b3b3b')
    # figure formatting
    fig1.legend(lines,[line.get_label() for line in lines],fontsize=12,loc='upper center',
                ncols=4,frameon=False,bbox_to_anchor=(0.5,0.97))
    
    ax1b[1].legend(fontsize=12,loc='upper center',ncols=4,frameon=False,bbox_to_anchor=(0.5,0.95),bbox_transform=fig1b.transFigure)
    
    fig1.suptitle('Impact of Varying Initial Conditions on Lotka-Volterra Predator-Prey Equations',
                  fontsize=16,fontweight='bold')
    
    fig1b.suptitle('Phase Diagrams for Lotka-Volterra Predator-Prey Equations',
                  fontsize=16,fontweight='bold')
    
    labkw = dict(size=16,weight='bold') # dictionary formats the axes labels to look nice
    ax1[2,1].set_xlabel(r'Time $\mathbf{(years)}$', **labkw)       # set common x label
    ax1[1,0].set_ylabel('Population/Carrying Capacity', **labkw)   # set common y label

    ax1b[1].set_xlabel(r'$\mathregular{N_1}$ (prey)', **labkw)       # set common x label
    ax1b[0].set_ylabel(r'$\mathregular{N_2}$ (predator)', **labkw)   # set common y label
    
    fig1.tight_layout()
    fig1b.tight_layout(w_pad=-5)
    
    # make directory to save plots if it doesn't already exist
    Path(f'{OUTPATH}').mkdir(parents=True, exist_ok=True)
    # save the figure
    fig1.savefig(f'{OUTPATH}/q3_panel.png',dpi=300)
    fig1b.savefig(f'{OUTPATH}/q3_phase.png',dpi=300)

def main():
    """
    executes the script fully as intended for the lab report
    """
    question_one()
    question_two()
    question_three()

# run the script
if __name__ == "__main__":
    main()

