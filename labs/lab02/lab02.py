#!/usr/bin/env python3
"""
lab02.py


"""
###########
# IMPORTS #
###########
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

# Changing font to stix; setting specialized math font properties as directly as possible
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.it'] = 'STIXGeneral:italic'
plt.rcParams['mathtext.bf'] = 'STIXGeneral:italic:bold'

plt.ion()  # comment out when not testing

#########################
# INTEGRATION FUNCTIONS #
#########################

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

    # integrate the function (could add a convergence check?)
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
    # Return values to caller.
    return time, N1, N2, [a,b,c,d]

def phase_plot(ax,data_dict):
    """
    Function creates phase plots to show the output of
    each ODE solver when solving the Lotka-Volterra equations.

    Function is intended to do all of the plotting and 
    formatting on a supplied axis, therefore note that the
    figure and axes need to be created outside of this helper
    function. This is intended to allow for maximum flexibility
    when making figures while reducing additional code.

    Input data should be formatted into a dictionary that follows the
    convention of storing output in this lab, which is as following

    data_dict = {
        'model'    : either 'Competition' or 'Predator-Prey',
        'params'   : [N, a, b, c, d],
        'times'    : array of times from a solver,
        'N1_euler' : array for equation N1 from the euler method,
        'N2_euler' : array for equation N2 from the euler method,
        'N1_rk8'   : array for equation N1 from the rk8 method,
        'N2_rk8'   : array for equation N2 from the rk8 method
    }

    Parameters
    ----------
    ax : mpl.Axes
        axis to create the phase plot on
    data_dict : dict
        dictionary containing all the information for the plot
    """
    # NOTE: we know the keys because the dictionary follows a standard specified format
    # first plot equation N1
    ax.plot(data_dict['times_euler'],data_dict['N1_euler'],
            lw=3,c='cornflowerblue',label=r'$\mathregular{N_1}$ Euler')
    
    ax.plot(data_dict['times_rk8'],data_dict['N1_rk8'],
            lw=3,c='cornflowerblue',label=r'$\mathregular{N_1}$ RK8',ls='--')
    
    # next plot equation N2
    ax.plot(data_dict['times_euler'],data_dict['N2_euler'],
            lw=3,c='indianred',label=r'$\mathregular{N_2}$ Euler')
    
    ax.plot(data_dict['times_rk8'],data_dict['N2_rk8'],
            lw=3,c='indianred',label=r'$\mathregular{N_2}$ RK8',ls='--')
    
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


    return ax

def solve_eqns(model='Competition',**kwargs):
    """
    Function solves Lotka-Volterra equations using both the euler and rk8 solvers
    and a set of defined parameters which are either taken from the defaults or
    passed in as kwargs. It then formats the data into a dictionary which is 
    used throughout the lab to store data.

    This dictionary takes the following form:
    data_dict = {
        'model'    : either 'Competition' or 'Predator-Prey',
        'params'   : [N, a, b, c, d],
        'times'    : array of times from a solver,
        'N1_euler' : array for equation N1 from the euler method,
        'N2_euler' : array for equation N2 from the euler method,
        'N1_rk8'   : array for equation N1 from the rk8 method,
        'N2_rk8'   : array for equation N2 from the rk8 method
    }

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

    The data dictionary takes the following form:
    data_dict = {
        'model'    : either 'Competition' or 'Predator-Prey',
        'params'   : [N, a, b, c, d],
        'times'    : array of times from a solver,
        'N1_euler' : array for equation N1 from the euler method,
        'N2_euler' : array for equation N2 from the euler method,
        'N1_rk8'   : array for equation N1 from the rk8 method,
        'N2_rk8'   : array for equation N2 from the rk8 method
    }

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
    data_dict : dict
        dictionary containing all data for the experiment in
        the standard format for the lab. This is intended for the
        user to explore.
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
    phase_plot(ax[0],competition_data)
    phase_plot(ax[1],pred_pray_data)
    
    # perform additional figure formatting (add params)
    return competition_data,pred_pray_data

def question_one():
    """
    Contains all necessary code to answer question 1 in the lab.

    
    """
    pass



unit_test()