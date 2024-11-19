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
two_levels_up = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add that directory to the system path
sys.path.append(two_levels_up)

from pyHMT2D import gVerbose
gVerbose=False

import LWD_module

plt.rc('text', usetex=True)  #allow the use of Latex for math expressions and equations
plt.rc('font', family='serif') #specify the default font family to be "serif"

if __name__ == "__main__":

    #Manning's n or Cd
    ManningN_or_Cd = "Cd"

    #read the calibrated parameter (Manning's n or Cd)
    calibrated_Cd_file = 'calibrated_Cd.dat'

    # Open the file and read the single float value
    with open(calibrated_Cd_file, 'r') as file:
        Cd = float(file.read().strip())  # Read and convert to float

    print("The calibrated Cd is {}".format(Cd))

    # run the case with the calibrated parameter
    LWD_module.run_srh_2d_model_with_given_Cd(Cd, destination_folder='case_final')

    print("All done!")