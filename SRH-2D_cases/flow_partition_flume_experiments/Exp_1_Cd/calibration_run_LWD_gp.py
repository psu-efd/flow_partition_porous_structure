import numpy as np
from matplotlib import pyplot as plt

from scipy.optimize import minimize
from skopt import gp_minimize, dump, load
from skopt.plots import plot_gaussian_process
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import Matern

import os
import sys

# Get the directory two levels above where LWD_module is
two_levels_up = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Add that directory to the system path
sys.path.append(two_levels_up)

from pyHMT2D import gVerbose
gVerbose=False

import ../LWD_module

plt.rc('text', usetex=True)  #allow the use of Latex for math expressions and equations
plt.rc('font', family='serif') #specify the default font family to be "serif"

def optimize_model_parameter(ManningN_or_Cd, parameter_bounds):

    # call the GP optimizer
    result = gp_minimize(
        objective_with_fixed_boolean,  # the function to minimize
        parameter_bounds,  # the bounds on each dimension of x
        acq_func="EI",  # the acquisition function
        n_calls=LWD_module.n_calls,  # the number of evaluations of f
        n_initial_points=LWD_module.n_initial_points,  # the number of random initialization points
        noise= (LWD_module.noise_level) ** 2,  # the noise variance of observation (5% estimated)
        random_state=1234)  # the random seed

    print("result = ", result)

    # save gp_minimize result
    dump(result, 'gp_result.pkl')

    # save simulation results to files: 'simulation_results.dat' and 'calibrated_ManningN.dat'/'calibrated_Cd.dat'
    LWD_module.save_calibration_results(ManningN_or_Cd, result.x[0])

# Define a wrapper function that only takes the continuous variable and uses the fixed boolean
def objective_with_fixed_boolean(x):
    # Whether it is full width or half-width (it is passed to the objective function)
    bFullWidth = False  #don't forget to change this value (full or half width)

    return LWD_module.objective_function_Cd(x, bFullWidth)

if __name__ == "__main__":

    #Manning's n or Cd
    ManningN_or_Cd = "Cd"

    # Define the bounds for the parameter Cd
    Cd_bounds = [(55.0, 70.0)]

    #optimize SRH-2D model parameter (Manning's n or Cd), and save result to "gp_result.pkl".
    optimize_model_parameter(ManningN_or_Cd, Cd_bounds)

    #plot optimization results
    LWD_module.plot_optimization_results(ManningN_or_Cd, Cd_bounds)

    print("All done!")