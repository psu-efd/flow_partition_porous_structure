#Solution for open channel hydraulics with a porous large woody debris (LWD)
#The solution is for the flow split (percentage of flow goes through the opening and LWD).

#The flow partition alpha=f(Fr, beta, Cd)

#The solutions of the simple model are saved in "Fr_beta_C_d_h2prime_h2_alpha_arrays_simple.npz"
#Externally, the

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from scipy.optimize import fsolve
from scipy.optimize import least_squares
from scipy.optimize import curve_fit

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import RFE

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
#from sklearn.inspection import plot_partial_dependence
import shap

from gplearn.genetic import SymbolicRegressor

from customized_SHAP_waterfall import custom_waterfall

from scipy.interpolate import griddata

import os
import json

import vtk
from vtk.util import numpy_support

plt.ioff()

plt.rc('text', usetex=True)  #allow the use of Latex for math expressions and equations
plt.rc('font', family='serif') #specify the default font family to be "serif"

#define some globle variables
Fr_min = 0.05   #lower bound for Fr
Fr_max = 0.95    #upper bound for Fr
Fr_n = 10        #number of values
beta_min = 0.05
beta_max = 0.95
beta_n = 10
C_d_min = 0.0
C_d_max = 80.0
C_d_n = 10

# Define the system of nonlinear equations for non-choking condition
def equations_not_choking(vars, Fr, beta, C_d, bLWDMomentumChange):
    h2prime, h2, alpha = vars

    eq1 = h2 + 1.0/2.0 * alpha**2/beta**2 / h2**2 * Fr**2 - (1 + 1.0/2.0*Fr**2)         # equation 1
    eq2 = h2prime**2 - h2**2 - C_d/h2 * ( (1-alpha)/(1-beta) )**2 * Fr**2                    # equation 2

    if bLWDMomentumChange: #if include the momentum change through LWD
        eq2 = eq2 + 2/h2prime* ((1-alpha)/(1-beta) )**2 * Fr**2 * (1-h2prime/h2)

    eq3 = h2prime + 1.0/2.0 * ((1-alpha)/(1-beta) )**2 * Fr**2 /h2prime**2 - (1 + 1.0/2.0*Fr**2)     # equation 3

    #print("residuals = ", eq1,eq2,eq3)

    return [eq1, eq2, eq3]

def jacobian_not_choking(vars, Fr, beta, C_d, bLWDMomentumChange):
    """
    Jacobian of equations_not_choking
    Parameters
    ----------
    vars
    Fr
    beta
    C_d
    bLWDMomentumChange

    Returns
    -------

    """

    h2prime, h2, alpha = vars

    jacobian = np.zeros([3, 3])

    jacobian[0, 0] = 0.0
    jacobian[0, 1] = 1.0 - alpha**2/beta**2/h2**3*Fr**2
    jacobian[0, 2] = alpha/beta**2/h2**2*Fr**2

    jacobian[1,0] = 2*h2prime
    jacobian[1,1] = -2*h2 +C_d/h2**2 *(1-alpha)**2/(1-beta)**2*Fr**2
    jacobian[1,2] = 2*C_d/h2*(1-alpha)/(1-beta)**2*Fr**2

    if bLWDMomentumChange:
        jacobian[1,0] += 2*(1-alpha)**2/(1-beta)**2*Fr**2*(1-h2prime/h2)
        jacobian[1,1] += 2*(1-alpha)**2/(1-beta)**2*Fr**2*h2prime**2/h2**2
        jacobian[1,2] -= 4/h2prime*(1-alpha)/(1-beta)**2*Fr**2*(1-h2prime/h2)

    jacobian[2,0] = 1 - (1-alpha)**2/(1-beta)**2*Fr**2/h2prime**3
    jacobian[2,1] = 0.0
    jacobian[2,2] = -(1-alpha)/(1-beta)**2*Fr**2/h2prime**2

    return jacobian

# Define the system of nonlinear equations for choking condition
def equations_choking(vars, Fr, beta, C_d, bLWDMomentumChange):
    h1star, h2prime, h2, alpha = vars

    eq1 = h1star + 1.0/2.0 * Fr**2 / h1star**2 - 3.0/2.0*(alpha/beta*Fr)**(2.0/3.0)  #equation 1

    eq2 = h1star + 1.0/2.0 * Fr**2 / h1star**2 - h2prime - 1.0/2.0 * (1-alpha)**2/(1-beta)**2 * Fr**2 / h2prime**2   # equation 2

    eq3 = h2prime**2 - h2**2 - C_d/h2 * ( (1-alpha)/(1-beta) )**2 * Fr**2                    # equation 3

    if bLWDMomentumChange: #if include the momentum change through LWD
        eq3 = eq3 + 2/h2prime* ((1-alpha)/(1-beta) )**2 * Fr**2 * (1-h2prime/h2)

    eq4 = h2prime -(alpha*Fr/beta)**(2.0/3.0)     # equation 4

    #print("residuals = ", eq1,eq2,eq3,eq4)

    return [eq1, eq2, eq3, eq4]

def jacobian_choking(vars, Fr, beta, C_d, bLWDMomentumChange):
    """
    Jacobian of equations_choking
    Parameters
    ----------
    vars
    Fr
    beta
    C_d
    bLWDMomentumChange

    Returns
    -------

    """

    h1star, h2prime, h2, alpha = vars

    jacobian = np.zeros([4, 4])

    jacobian[0, 0] = 1 - Fr**2/h1star**3
    jacobian[0, 1] = 0
    jacobian[0, 2] = 0
    jacobian[0, 3] = -1/alpha**(1.0/3.0)*(Fr/beta)**(2.0/3.0)

    jacobian[1, 0] = 1 - Fr**2/h1star**3
    jacobian[1, 1] = -1 + (1-alpha)**2/(1-beta)**2*Fr**2/h2prime**3
    jacobian[1, 2] = 0
    jacobian[1, 3] = (1-alpha)/(1-beta)**2*Fr**2/h2prime**2

    jacobian[2,0] = 0
    jacobian[2,1] = 2*h2prime
    jacobian[2,2] = -2*h2 +C_d/h2**2 *(1-alpha)**2/(1-beta)**2*Fr**2
    jacobian[2,3] = 2*C_d/h2*(1-alpha)/(1-beta)**2*Fr**2

    if bLWDMomentumChange:
        jacobian[2,1] += 2*(1-alpha)**2/(1-beta)**2*Fr**2*(1-h2prime/h2)
        jacobian[2,2] += 2*(1-alpha)**2/(1-beta)**2*Fr**2*h2prime**2/h2**2
        jacobian[2,3] -= 4/h2prime*(1-alpha)/(1-beta)**2*Fr**2*(1-h2prime/h2)

    jacobian[3,0] = 0
    jacobian[3,1] = 0
    jacobian[3,2] = 1
    jacobian[3,3] = -2.0/3.0/alpha**(1.0/3.0)*(Fr/beta)**(2.0/3.0)

    return jacobian


def solve_LWD_for_given_Fr_beta_C_d(Fr, beta, C_d, bLWDMomentumChange):
    """
    Solve the flow around porous LWD problem for a given set of parameters. The strategy is to first try the not-choking
    condition. If there is no solution, then the flow must be choked.

    Parameters
    ----------

    Returns
    -------

    """

    #Try to solve with fsolve first. This flag controls whether to do additional
    #solve with lease squares if fsolve fails.
    bAdditionalSolve_with_least_squares = True

    #a flag for whether the flow is choked
    bChoked = 0  #0-not choked; 1-choked
    bConverged = 1 #1-converged (found a good solution); 0-diverged

    print("Fr, beta, C_d, bLWDMomentumChange=", Fr, beta, C_d, bLWDMomentumChange)

    # if solver_option==1:      #solve with fsolve
    #try to solve with fsolve first.
    print("    Solving with fsolve: assuming not choked condition...")

    #try with the not-choking condition first to see whether it works. In not-choking condition,
    #there are three unknows and three equations
    # unknowns: h2prime, h2, alpha
    initial_guess = [1.0, 1.0, 0.5]

    #solution, infodict, ier, mesg = fsolve(equations_not_choking, initial_guess, args=(Fr,beta,C_d,bLWDMomentumChange), full_output=True)
    solution, infodict, ier, mesg = fsolve(equations_not_choking, initial_guess, args=(Fr, beta, C_d, bLWDMomentumChange), fprime=jacobian_not_choking, full_output=True)

    if ier==1:
        bConverged = 1
    else:
        bConverged = 0

    # Display the solution
    print(f"        Solution: h2prime = {solution[0]}, h2 = {solution[1]}, alpha = {solution[2]}")
    # print("solution = ", solution)
    #print("    ier=", ier)
    # print("mesg=",mesg)
    #print("infodict=",infodict)
    # print("residuals=",np.isclose(equations_not_choking(solution,Fr,beta,C_d,bLWDMomentumChange),[0,0,0]))
    #print("residuals=", equations_not_choking(solution, Fr, beta, C_d, bLWDMomentumChange))

    # check positivity of solution
    if solution[0] < 0.0 or solution[1] < 0.0 or solution[2] < 0.0:
        bConverged = 0

    #if the solution is not converged, assume the flow is choked (no solution)
    if bConverged==0:
        print("        The flow may be choked. Solve with the choking condition.")

        # unknowns: h1star, h2prime, h2, alpha
        initial_guess = [1.5, 1.1, 0.8, 0.8]

        #solution, infodict, ier, mesg = fsolve(equations_choking, initial_guess, args=(Fr, beta, C_d, bLWDMomentumChange), full_output=True)
        solution, infodict, ier, mesg = fsolve(equations_choking, initial_guess, args=(Fr, beta, C_d, bLWDMomentumChange), fprime=jacobian_choking, full_output=True)

        if ier == 1:
            bConverged = 1
            bChoked = 1     #if converged, then yes, it is a choked flow.
        else:
            bConverged = 0
            bChoked = 0     #if not conerged, then it has nothing to do with the choked flow.

        # Display the solution
        print(f"        Solution: h1star = {solution[0]}, h2prime = {solution[1]}, h2 = {solution[2]}, alpha = {solution[3]}")
        # print("solution = ", solution)
        #print("    ier=", ier)
        #print("    mesg=",mesg)
        #print("    infodict=",infodict)
        #print("    residuals=",np.isclose(equations_choking(solution,Fr,beta,C_d,bLWDMomentumChange),[0,0,0,0]))
        #print("    residuals=", equations_choking(solution, Fr, beta, C_d, bLWDMomentumChange))

        # check positivity of solution
        if solution[0] < 0.0 or solution[1] < 0.0 or solution[2] < 0.0:
            bConverged = 0
            bChoked = 0  # if not conerged, then it has nothing to do with the choked flow.


    #elif solver_option==2:     #solve with least_squares
    if (not bConverged) and bAdditionalSolve_with_least_squares:  #if not converged with fsolve, try again with least_squares

        # unknowns: h2prime, h2, alpha
        initial_guess = [1.0, 1.0, 0.5]

        # Define bounds for the variables [h2prime, h2, alpha]
        # Lower bounds
        lower_bounds = [0.01, 0.01, 0]
        # Upper bounds
        upper_bounds = [3.0, 2.0, 1.0]  # assume the water depht upper bound is two times the upstream water depth

        result = least_squares(equations_not_choking, initial_guess, bounds=(lower_bounds, upper_bounds),
                               args=(Fr,beta,C_d,bLWDMomentumChange),
                               jac=jacobian_not_choking,
                               method='dogbox')
        # Extract the solution
        solution = result.x

        #print("residuals=",equations_not_choking(solution, Fr, beta, C_d, bLWDMomentumChange))
        #print("residuals=", np.isclose(equations_not_choking(solution, Fr, beta, C_d, bLWDMomentumChange), [0, 0, 0]))

        residuals = np.isclose(equations_not_choking(solution, Fr, beta, C_d, bLWDMomentumChange), [0, 0, 0])

        # Check if the optimization was successful
        if result.success:
            bConverged = 1
            # Display the solution
            print("Fr, beta, C_d, bLWDMomentumChange=", Fr, beta, C_d, bLWDMomentumChange)
            print(f"Solution: h2prime = {solution[0]}, h2 = {solution[1]}, alpha = {solution[2]}")
        else:
            bConverged = 0
            print("Optimization failed:", result.message)
            print("Fr, beta, C_d, bLWDMomentumChange=", Fr, beta, C_d, bLWDMomentumChange)
            print("result = ", result)
            #exit()

        #check positivity of solution
        if solution[0] < 0.0 or solution[1] < 0.0 or solution[2] < 0.0 or solution[2] > 1.0:
            bConverged = 0

        if bConverged==0:  #flow must be choked
            print("    The flow is choked. Solve with the choking condition.")

            # unknowns: h1star, h2prime, h2, alpha
            initial_guess = [1.1, 1.1, 1.0, 0.3]

            # Define bounds for the variables [h1star, h2prime, h2, alpha]
            # Lower bounds
            lower_bounds = [1.0, 0.01, 0.01, 0]
            # Upper bounds
            upper_bounds = [3.0, 2.0, 2.0, 1.0]  # assume the water depht upper bound is two times the upstream water depth

            result = least_squares(equations_choking, initial_guess, bounds=(lower_bounds, upper_bounds),
                                   args=(Fr, beta, C_d, bLWDMomentumChange),
                                   jac=jacobian_choking, method='dogbox')
            # Extract the solution
            solution = result.x

            # Check if the optimization was successful
            if result.success:
                bConverged = 1
                bChoked = 1

                # Display the solution
                print("    Fr, beta, C_d, bLWDMomentumChange=", Fr, beta, C_d, bLWDMomentumChange)
                print(f"    Solution: h1star= {solution[0]},h2prime = {solution[1]}, h2 = {solution[2]}, alpha = {solution[3]}")
            else:
                bConverged = 0
                bChoked = 0
                print("    Optimization failed:", result.message)
                print("    Fr, beta, C_d, bLWDMomentumChange=", Fr, beta, C_d, bLWDMomentumChange)
                print("    result = ", result)

            # check positivity of solution
            if solution[0] < 0.0 or solution[1] < 0.0 or solution[2] < 0.0 or solution[3] < 0.0 or solution[3] > 1.0:
                bConverged = 0

            print("    residuals=", equations_choking(solution, Fr, beta, C_d, bLWDMomentumChange))
            print("    residuals=",
                  np.isclose(equations_choking(solution, Fr, beta, C_d, bLWDMomentumChange), [0, 0, 0, 0]))

            residuals = np.isclose(equations_choking(solution, Fr, beta, C_d, bLWDMomentumChange), [0, 0, 0, 0])

        #print("result = ", result)

    return solution, bChoked, bConverged

def save_3D_array_to_vtk(array_3d, var_name, vtk_filename):
    """
    Save a 3D array to a VTK file.

    Parameters
    ----------
    array_3d
    var_name: str
    vtk_filename

    Returns
    -------

    """

    # Create a vtkImageData object to store the 3D NumPy array
    vtk_data = vtk.vtkImageData()
    vtk_data.SetDimensions(array_3d.shape)

    # Set the origin and spacing (optional, depending on your specific data)
    vtk_data.SetOrigin(0, 0, 0)
    vtk_data.SetSpacing(1, 1, 1)

    # Convert the NumPy array to a VTK array
    vtk_array = numpy_support.numpy_to_vtk(num_array=array_3d.ravel(), deep=True, array_type=vtk.VTK_FLOAT)

    # Specify the name of the variable
    vtk_array.SetName(var_name)  # Set the name for the variable

    # Assign the VTK array to the VTK image data object
    vtk_data.GetPointData().SetScalars(vtk_array)

    # Create a VTK XML image data writer (for .vti format)
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(vtk_filename)
    writer.SetInputData(vtk_data)

    # Write the file
    writer.Write()

    print("3D array saved to VTK file ", vtk_filename)

def solve_LWD():
    """
    Solve the simple model for flow partition problem for the porous structure (LWD) in open channel flow

    Returns
    -------

    """

    #define samples on parameter space
    #upstream Froude number
    Frs = np.linspace(Fr_min, Fr_max, Fr_n)
    #opening width fraction at LWD
    betas = np.linspace(beta_min, beta_max, beta_n)
    #LWD dimensionless drag parameter
    C_ds = np.linspace(C_d_min, C_d_max, C_d_n)

    #whether to consider the momentum change within LWD
    bLWDMomentumChange = True

    # Initialize an empty 3D data cubes to store results
    iCases_results = np.empty((len(Frs), len(betas), len(C_ds)))
    h2prime_results = np.empty((len(Frs), len(betas), len(C_ds)))
    h2_results = np.empty((len(Frs), len(betas), len(C_ds)))
    alpha_results = np.empty((len(Frs), len(betas), len(C_ds)))
    bChoked_results = np.empty((len(Frs), len(betas), len(C_ds)))
    bConverged_results = np.empty((len(Frs), len(betas), len(C_ds)))

    # Loop over each combination of Fr, beta, C_d values
    iCase = 0
    for i in range(len(Frs)):  # Loop over Fr-values
        for j in range(len(betas)):  # Loop over beta-values
            for k in range(len(C_ds)):  # Loop over C_d-values
                # Extract the current values of Fr, beta, C_d
                Fr = Frs[i]
                beta = betas[j]
                C_d = C_ds[k]

                # Perform some computation using Fr, beta, and C_d
                solution, bChoked, bConverged = solve_LWD_for_given_Fr_beta_C_d(Fr, beta, C_d, bLWDMomentumChange)

                # Store the results in the data cubes
                if not bChoked:
                    h2prime_results[i, j, k] = solution[0]
                    h2_results[i, j, k] = solution[1]
                    alpha_results[i, j, k] = solution[2]
                    bChoked_results[i, j, k] = bChoked
                    bConverged_results[i, j, k] = bConverged
                else:
                    h2prime_results[i, j, k] = solution[1]
                    h2_results[i, j, k] = solution[2]
                    alpha_results[i, j, k] = solution[3]
                    bChoked_results[i, j, k] = bChoked
                    bConverged_results[i, j, k] = bConverged

                iCases_results[i, j, k] = iCase

                iCase += 1

    #save results to files
    save_3D_array_to_vtk(h2prime_results, "h2prime", "h2prime_results.vti")
    save_3D_array_to_vtk(h2_results, "h2", "h2_results.vti")
    save_3D_array_to_vtk(alpha_results, "alpha", "alpha_results.vti")
    save_3D_array_to_vtk(bChoked_results, "bChoked", "bChoked_results.vti")

    # Save arrays in a .npz file (compressed)
    print("Saving Fr_beta_C_d_h2prime_h2_alpha_arrays_simple.npz")
    np.savez_compressed('Fr_beta_C_d_h2prime_h2_alpha_arrays_simple.npz',
                        iCases=iCases_results,
                        Frs=Frs, betas=betas, C_ds=C_ds,
                        h2prime=h2prime_results, h2=h2_results, alpha=alpha_results,
                        bChoked=bChoked_results, bConverged=bConverged_results)

def plot_alpha_distributions_simple_model_vs_SRH_2D(alphas_1D_simple_converged, alphas_1D_SRH_2D_converged):
    """
    Plot alpha value distributions from simple model and SRH-2D
    Parameters
    ----------
    alphas_1D_simple_converged
    alphas_1D_SRH_2D_converged

    Returns
    -------

    """

    # Define the number of bins
    num_bins = 10

    # Creating subplots with multiple histograms
    fig, ax = plt.subplots(figsize=(6, 6))

    # peak_easement_exceedance_surcharge
    counts, bin_edges = np.histogram(alphas_1D_simple_converged, bins=num_bins)
    # Calculate bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    # Calculate the mean, median, min and max values
    mean_value = np.mean(alphas_1D_simple_converged)
    median_value = np.median(alphas_1D_simple_converged)
    min_value = np.min(alphas_1D_simple_converged)
    max_value = np.max(alphas_1D_simple_converged)

    #ax.bar(bin_centers, counts, width=bin_edges[1] - bin_edges[0], color='blue', alpha=0.7, edgecolor='k')
    ax.hist(alphas_1D_simple_converged, bins=num_bins, label='Simple model', color='blue', alpha=0.7, edgecolor='k')

    ax.text(0.01, 300,
                     f'Simple model:\n\tMedian: {median_value:.2f} \n\tMean: {mean_value:.2f} ',
                     # transform=plt.gca().transAxes,
                     color='black',
                     fontsize=14,
                     horizontalalignment='left',
                     verticalalignment='top'
                     )

    # maximum_easement_exceedance_surcharge
    counts, bin_edges = np.histogram(alphas_1D_SRH_2D_converged, bins=num_bins)
    # Calculate bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    # Calculate the mean, median, min and max values
    mean_value = np.mean(alphas_1D_SRH_2D_converged)
    median_value = np.median(alphas_1D_SRH_2D_converged)
    min_value = np.min(alphas_1D_SRH_2D_converged)
    max_value = np.max(alphas_1D_SRH_2D_converged)

    #ax.bar(bin_centers, counts, width=bin_edges[1] - bin_edges[0], color='red', alpha=0.7, edgecolor='k')
    ax.hist(alphas_1D_SRH_2D_converged, bins=num_bins, label='SRH-2D', color='red', alpha=0.7, edgecolor='k')

    ax.text(0.01, 250,
                     f'SRH-2D: \n\tMedian: {median_value:.2f} \n\tMean: {mean_value:.2f} ',
                     # transform=plt.gca().transAxes,
                     color='black',
                     fontsize=12,
                     horizontalalignment='left',
                     verticalalignment='top'
                     )


    # Adding labels and title
    ax.set_xlabel(r'Flow partition $\alpha$', fontsize=18)
    ax.set_ylabel('Count', fontsize=18)
    #ax.set_ylim(0, 110)

    # Set font size of axis ticks
    ax.tick_params(axis='both', which='major', labelsize=14)  # Adjust labelsize as needed

    # Adjusting layout for better spacing
    # plt.tight_layout()

    ax.legend(
        loc="upper left",          # Position of the legend
        fontsize=14,                # Font size of the legend text
        frameon=True,               # Add a frame around the legend
        fancybox=False,              # Rounded edges on the legend box
        shadow=False,                # Add a shadow to the legend box
        #title="Legend Title",       # Title for the legend
        #title_fontsize=12           # Font size for the legend title
    )

    figureFileName = "alpha_distributions_simple_model_vs_SRH_2D.png"
    fig.savefig(figureFileName, dpi=300, bbox_inches='tight', pad_inches=0.1)

    # Display the figure
    plt.show()

def plot_alpha_diff_distributions_simple_model(alphas_1D_diff):
    """
    Plot alpha_diff value distributions: simple model - SRH-2D
    Parameters
    ----------
    alphas_1D_diff

    Returns
    -------

    """

    # Define the number of bins
    num_bins = 10

    # Creating subplots with multiple histograms
    fig, ax = plt.subplots(figsize=(6, 6))

    # peak_easement_exceedance_surcharge
    counts, bin_edges = np.histogram(alphas_1D_diff, bins=num_bins)
    # Calculate bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    # Calculate the mean, median, min and max values
    mean_value = np.mean(alphas_1D_diff)
    median_value = np.median(alphas_1D_diff)
    min_value = np.min(alphas_1D_diff)
    max_value = np.max(alphas_1D_diff)

    #ax.bar(bin_centers, counts, width=bin_edges[1] - bin_edges[0], color='blue', alpha=0.7, edgecolor='k')
    ax.hist(alphas_1D_diff, bins=num_bins, label='Simple model', color='blue', alpha=0.7, edgecolor='k')

    ax.text(0.6, 600,
                     f'Median: {median_value:.2f} \n\tMean: {mean_value:.2f} ',
                     # transform=plt.gca().transAxes,
                     color='black',
                     fontsize=14,
                     horizontalalignment='left',
                     verticalalignment='top'
                     )

    # Adding labels and title
    ax.set_xlabel(r'$\alpha_{error}$ due to the simple model', fontsize=16)
    ax.set_ylabel('Count', fontsize=16)
    #ax.set_ylim(0, 110)

    # Set font size of axis ticks
    ax.tick_params(axis='both', which='major', labelsize=14)  # Adjust label size as needed

    # Adjusting layout for better spacing
    plt.tight_layout()

    figureFileName = "alpha_diff_distributions_simple_model.png"
    fig.savefig(figureFileName, dpi=300, bbox_inches='tight', pad_inches=0)

    # Display the figure
    plt.show()

def plot_alpha_random_forest_regression_vs_truth(alpha_random_forest, alpha_truth, rmse, r2_score, Fr, beta, Cd):
    """
    Plot and compare alpha values from random forest regression model and truth
    Parameters
    ----------
    alpha_random_forest
    alpha_truth

    Returns
    -------

    """

    # make plot
    fig, ax = plt.subplots(figsize=(6, 6))

    # Define a colormap (e.g., 'coolwarm' colormap)
    cmap = plt.get_cmap('coolwarm')

    # Plot the diagonal line
    ax.plot([0, 1], [0, 1], color='black', linestyle='--')

    color_var = None

    #scatter_plot = ax.scatter(alpha_random_forest, alpha_truth, facecolors='none', edgecolor='black', alpha=0.5, marker='o', s=60)  # s=60
    scatter_plot = ax.scatter(alpha_random_forest, alpha_truth, facecolors='none', edgecolor='black', marker='o', s=60)  # s=60
    ax.text(0.05, 0.8, f"RMSE = {rmse:.3f}", fontsize=16, color="black", ha="left")
    ax.text(0.05, 0.75, f"$r^2$ = {r2_score:.3f}", fontsize=16, color="black", ha="left")

    cbar = plt.colorbar(scatter_plot, shrink=1.0, fraction=0.03, pad=0.04, aspect=40)
    # Customize colorbar
    #cbar.set_label(r"$C_d$", fontsize=16)  # Set colorbar title font size
    #cbar.ax.tick_params(labelsize=14)  # Set colorbar label number font size

    # Hide the colorbar
    cbar.ax.set_visible(False)

    # set the limit for the x and y axes
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    ax.set_aspect('equal')

    # set x and y axes label and font size
    ax.set_xlabel(r'$\alpha$ from random forest regression', fontsize=18)
    ax.set_ylabel(r'$\alpha$ from SRH-2D', fontsize=18)

    # show the ticks on both axes and set the font size
    ax.tick_params(axis='both', which='major', labelsize=16)
    # xtick_spacing=1
    # ax.xaxis.set_major_locator(tick.MultipleLocator(xtick_spacing))
    # ax.xaxis.set_major_formatter(tick.FormatStrFormatter('%.1f'))

    # show legend, set its location, font size, and turn off the frame
    # plt.legend(loc='lower right', fontsize=14, frameon=False)

    fig.savefig("alpha_comparison_random_forest_vs_SRH_2D.png", dpi=300, bbox_inches='tight',
                pad_inches=0)

    plt.show()

    #plt.close()

def plot_alpha_diff_random_forest_regression_vs_truth(alpha_diff_random_forest, alpha_diff_truth, Fr, beta, Cd):
    """
    Plot and compare alpha diff values from random forest regression model and truth
    Parameters
    ----------

    Returns
    -------

    """

    rmse = np.sqrt(np.mean((alpha_diff_random_forest - alpha_diff_truth) ** 2))
    r2 = r2_score(alpha_diff_random_forest, alpha_diff_truth)

    print("rmse = ", rmse)
    print("r2 = ", r2)

    color_variable_names = ["none", "Fr", "beta", "Cd"]

    for iPlot in range(4):
        color_variable_name = color_variable_names[iPlot]

        # make plot
        fig, ax = plt.subplots(figsize=(6, 6))

        # Plot the diagonal line
        ax.plot([0, 0.5], [0, 0.5], color='black', linestyle='--')

        if color_variable_name=="none":
            scatter_plot = ax.scatter(alpha_diff_random_forest, alpha_diff_truth, facecolors='none', edgecolor='black',
                                      marker='o', s=60)  # s=60
        elif color_variable_name=="Fr":
            scatter_plot = ax.scatter(alpha_diff_random_forest, alpha_diff_truth, c=Fr,
                                      vmin=0, vmax=1, cmap="coolwarm",
                                      marker='o', s=60)  # s=60
        elif color_variable_name=="beta":
            scatter_plot = ax.scatter(alpha_diff_random_forest, alpha_diff_truth, c=beta,
                                      vmin=0, vmax=1, cmap="coolwarm",
                                      marker='o', s=60)  # s=60
        elif color_variable_name=="Cd":
            scatter_plot = ax.scatter(alpha_diff_random_forest, alpha_diff_truth, c=Cd,
                                      vmin=0, vmax=80, cmap="coolwarm",
                                      marker='o', s=60)  # s=60

        if color_variable_name!="none":
            cbar = plt.colorbar(scatter_plot, shrink=1.0, fraction=0.03, pad=0.04, aspect=40)
        if color_variable_name == "Fr":
            cbar.set_label(r'$Fr$', fontsize=16)
        elif color_variable_name == "beta":
            cbar.set_label(r'$\beta$', fontsize=16)
        elif color_variable_name == "Cd":
            cbar.set_label(r'$C_d$', fontsize=16)

        if color_variable_name!="none":
            cbar.ax.tick_params(labelsize=14)  # Set tick font size

            # Additional customizations if needed
            #cbar.outline.set_linewidth(0.5)  # Adjust colorbar outline width

        ax.text(0.1, 0.4, f"RMSE = {rmse:.3f}", fontsize=14, color="black", ha="left")
        ax.text(0.1, 0.35, f"$r^2$ = {r2:.3f}", fontsize=14, color="black", ha="left")

        # set the limit for the x and y axes
        ax.set_xlim([0, 0.5])
        ax.set_ylim([0, 0.5])

        ax.set_aspect('equal')

        # set x and y axes label and font size
        ax.set_xlabel(r'$\alpha_{error}$ from random forest regression', fontsize=16)
        ax.set_ylabel(r'$\alpha_{error}$ truth values', fontsize=16)

        # show the ticks on both axes and set the font size
        ax.tick_params(axis='both', which='major', labelsize=14)
        # xtick_spacing=1
        # ax.xaxis.set_major_locator(tick.MultipleLocator(xtick_spacing))
        # ax.xaxis.set_major_formatter(tick.FormatStrFormatter('%.1f'))

        # show legend, set its location, font size, and turn off the frame
        # plt.legend(loc='lower right', fontsize=14, frameon=False)

        fig.savefig("alpha_diff_comparison_random_forest_vs_truth_"+color_variable_names[iPlot]+".png",
                    dpi=300, bbox_inches='tight', pad_inches=0.0)

        plt.show()

        #plt.close()

def using_SHAP(Fr, beta, Cd, var, feature_names):
    """
    Using SHAP (SHapley Additive exPlanations) for Feature Importance.
    This is global analysis across the whole dataset.
    Random Forest Regressor is still used.

    :param Fr:
    :param beta:
    :param Cd:
    :param alpha_diff:
    :return:
    """

    X = np.column_stack((Fr, beta, Cd))
    alpha_truth = var

    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=feature_names)  # Update with actual names

    #prepare training and test dataset
    X_train, X_test, y_train, y_test = train_test_split(X, alpha_truth, test_size=0.2, random_state=42)

    #create and train the model
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    #make prediction on the whole dataset
    alpha_pred = model.predict(X)

    # rmse and R2 score using the model's score method
    rmse = np.sqrt(mean_squared_error(alpha_truth, alpha_pred))
    r2_score = model.score(X, alpha_truth)
    print("RMSE:", rmse)
    print("R2 Score:", r2_score)

    #plot comparison of alpha between random forest regression and SRH-2D
    plot_alpha_random_forest_regression_vs_truth(alpha_pred, alpha_truth, rmse, r2_score, Fr, beta, Cd)

    # Use SHAP to explain predictions
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Convert SHAP values and feature data to a DataFrame for easier plotting
    shap_values_df = pd.DataFrame(shap_values, columns=X.columns)
    mean_shap_values = np.abs(shap_values_df).mean().sort_values(ascending=False)  # Sort by importance

    # Dependence plot
    bPlot_alpha_SHAP_values_for_features = True
    if bPlot_alpha_SHAP_values_for_features:
        shap_values_temp = explainer(X)

        # Optionally, color the plot by another feature
        #shap.dependence_plot(r"$Fr$", shap_values_temp.values, X, interaction_index=r"$\beta$") # Color by a feature
        #shap.dependence_plot(r"$\beta$", shap_values_temp.values, X, interaction_index=r"$C_d$")  # Color by a feature
        #shap.dependence_plot(r"$C_d$", shap_values_temp.values, X, interaction_index=r"$\beta$")  # Color by a feature

        # Define features to plot
        features_to_plot = [r"$Fr$", r"$\beta$", r"$C_d$"]  # Specify the features you want to plot
        features_to_color = [r"$\beta$", r"$C_d$", r"$\beta$"]  # Specify the features you want to plot
        #features_to_color = [None, None, None]  # Specify the features you want to plot
        feature_names_text = ["Fr", "beta", "Cd"]
        #feature_names = [r"$Fr$", r"$\beta$", r"$C_d$"]

        # Determine the y-axis range by finding the min and max SHAP values across all features
        y_min = np.min(shap_values_temp.values)
        y_max = np.max(shap_values_temp.values)

        for iPlot in range(3):
            # Set up subplots
            fig, ax = plt.subplots(figsize=(6, 6))

            #shap.dependence_plot(features_to_plot[iPlot], shap_values_temp.values, X, interaction_index=features_to_color[iPlot], cmap='coolwarm', ax=ax, show=False)
            if iPlot == 0:  #Fr colored with beta
                scatter_plot = ax.scatter(Fr, shap_values_temp.values[:,0], c=beta, vmin=0, vmax=1.0, cmap='coolwarm', marker='o', s=60)

                cbar = plt.colorbar(scatter_plot, shrink=1.0, fraction=0.03, pad=0.04, aspect=40)
                # Customize colorbar
                cbar.set_label(r"$\beta$", fontsize=18)  # Set colorbar title font size
                cbar.ax.tick_params(labelsize=16)  # Set colorbar label number font size

            elif iPlot == 1:  #beta color with Cd
                scatter_plot = ax.scatter(beta, shap_values_temp.values[:,1], c=Cd, vmin=0, vmax=80, cmap='coolwarm', marker='o', s=60)

                cbar = plt.colorbar(scatter_plot, shrink=1.0, fraction=0.03, pad=0.04, aspect=40)
                # Customize colorbar
                cbar.set_label(r"$C_d$", fontsize=18)  # Set colorbar title font size
                cbar.ax.tick_params(labelsize=16)  # Set colorbar label number font size

            elif iPlot == 2:  #Cd colored with beta
                scatter_plot = ax.scatter(Cd, shap_values_temp.values[:,2], c=beta, vmin=0, vmax=1.0, cmap='coolwarm', marker='o', s=60)

                cbar = plt.colorbar(scatter_plot, shrink=1.0, fraction=0.03, pad=0.04, aspect=40)
                # Customize colorbar
                cbar.set_label(r"$\beta$", fontsize=18)  # Set colorbar title font size
                cbar.ax.tick_params(labelsize=16)  # Set colorbar label number font size

            ax.set_ylim(y_min, y_max)  # Set y-axis limits to be the same across plots

            if feature_names[iPlot] == "Fr":
                ax.set_xlim(-0.01, 1.01)
            elif feature_names[iPlot] == "beta":
                ax.set_xlim(-0.01, 1.01)
            elif feature_names[iPlot] == "Cd":
                ax.set_xlim(-1, 81)

            ax.tick_params(axis='both', which='major', labelsize=16)
            ax.set_xlabel(features_to_plot[iPlot], fontsize=18)
            ax.set_ylabel("SHAP value for "+ features_to_plot[iPlot], fontsize=18)

            # Show only the top and bottom borders (spines)
            ax.spines['top'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_visible(True)
            ax.spines['right'].set_visible(True)

            # Display the plot
            plt.tight_layout()

            plt.savefig("alpha_dependence_plot_SHAP_SRH_2D_"+feature_names_text[iPlot]+".png", dpi=300, bbox_inches='tight', pad_inches=0.05)

            plt.show()

    # Plot a custom summary bar plot for SHAP values
    bPlot_alpha_SHAP_bar=True
    if bPlot_alpha_SHAP_bar:
        plt.figure(figsize=(6, 4))
        plt.barh(mean_shap_values.index, mean_shap_values.values, color="deepskyblue")

        plt.xlabel(r"SHAP value (impact on flow partition $\alpha$)", fontsize=18)
        plt.ylabel("Variables", fontsize=18)
        #plt.title("Customized SHAP Summary Plot", fontsize=16, fontweight="bold")
        plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature on top
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid(axis="x", linestyle="--", alpha=0.7)

        plt.xlim(0,0.25)

        # Add text for each bar
        for index, value in enumerate(mean_shap_values):
            plt.text(value, index, f'{value:.3f}', fontsize=16, va='center')  # `value` sets the x-position, `index` sets the y-position

        plt.savefig("alpha_dependence_SHAP_bar_SRH_2D.png", dpi=300, bbox_inches='tight', pad_inches=0)

        plt.show()

    # Plot a custom summary bar plot for SHAP values
    bPlot_alpha_SHAP_summary = True
    if bPlot_alpha_SHAP_summary:
        # Set up subplots
        fig, ax = plt.subplots(figsize=(8, 6))

        shap.summary_plot(shap_values, X, feature_names=feature_names, cmap="coolwarm", show=False)

        # Customizing the plot
        # Access the current figure and axis after the plot is created
        fig = plt.gcf()  # Get the current figure
        ax = plt.gca()  # Get the current axis

        # plt.title("Customized SHAP Summary Plot", fontsize=18, fontweight='bold')
        ax.set_xlabel(r"SHAP value (impact on flow partition $\alpha$)", fontsize=18)
        ax.set_ylabel("Variables", fontsize=18)

        # Customize the color bar legend
        cbar = plt.gcf().axes[-1]  # Access the last axis, which is the color bar in SHAP plot
        cbar.set_ylabel("Variable value", fontsize=16)  # Change "Feature value" to "Variable value"

        # Customize tick labels
        ax.tick_params(axis='x', labelsize=16)
        # ax.tick_params(axis='y', labelsize=12)

        # Add grid lines (optional)
        # plt.grid(True, linestyle='--', alpha=0.5)

        # Show the customized plot
        # plt.tight_layout()

        fig.savefig("alpha_dependence_SHAP_summary_SRH_2D.png", dpi=300, bbox_inches='tight', pad_inches=0)

        plt.show()

def using_SHAP_local(Fr, beta, Cd, var, feature_names):
    """
    Using SHAP (SHapley Additive exPlanations) for Feature Importance. This is local analysis at
    selected predictions.

    Random Forest Regressor is still used.

    :param Fr:
    :param beta:
    :param Cd:
    :param var=alpha:
    :return:
    """

    X = np.column_stack((Fr, beta, Cd))
    y = var

    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=feature_names)  # Update with actual names

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    y_pred = model.predict(X)

    # R2 score using the model's score method
    r2_score = model.score(X_test, y_test)
    print("R2 Score:", r2_score)

    # Choose a sample to explain
    #case ID: Fr, beta, Cd, alpha
    #1: 0.05,0.05,8.89, 0.11
    #139: 0.15,0.35,80.00, 0.77
    #353: 0.35,0.55,26.67, 0.82
    #772: 0.75,0.75,17.78, 0.90

    sample_idxs = [1, 13, 642, 772]

    for sample_idx in sample_idxs:
        print("case ID, Fr, beta, Cd, alpha:", sample_idx, Fr[sample_idx], beta[sample_idx], Cd[sample_idx], var[sample_idx])

        X_sample = X.iloc[[sample_idx]]

        # Calculate SHAP values
        explainer = shap.TreeExplainer(model, X)
        shap_values = explainer(X_sample)

        print("explainer.expected_value, np.mean(y)=", explainer.expected_value, np.mean(y))

        custom_feature_names = [r"$Fr$", r"$\beta$", r"$C_d$"]

        # Calculate the prediction value for the sample
        prediction_value = shap_values[0].base_values + shap_values[0].values.sum()

        # Define custom text for the prediction
        custom_prediction_text = r"$\alpha(Fr, \beta, C_d) = $" + f"{prediction_value:.2f}"

        # Create a new Explanation object with custom labels but default order
        custom_shap_values = shap.Explanation(
            values=shap_values[0].values,
            base_values=shap_values[0].base_values,
            data=shap_values[0].data,
            feature_names=custom_feature_names
        )

        # Plot waterfall plot with reordered features
        custom_waterfall(custom_shap_values, show=False)

        # Access all axes in the current figure
        axes = plt.gcf().get_axes()

        # If ax3 is the third axis, it should be at index 2
        if len(axes) >= 3:
            ax1 = axes[0]
            ax2 = axes[1]
            ax3 = axes[2]

            # You can now modify ax3, for example, changing the title or text
            ax1.set_xlim(0, 1)
            ax2.set_xlim(0, 1)
            ax3.set_xlim(0, 1)

        plt.savefig("alpha_local_dependence_SHAP_SRH_2D_case_" + str(sample_idx).zfill(4) + ".png", dpi=300, bbox_inches='tight', pad_inches=0)

        plt.show()

def using_SHAP_for_alpha_diff(Fr, beta, Cd, var, feature_names):
    """
    Using SHAP (SHapley Additive exPlanations) for Feature Importance. Random Forest Regressor
     is still used. This function concerns alpha_diff=f(Fr, beta, Cd)

    :param Fr:
    :param beta:
    :param Cd:
    :param alpha_diff:
    :return:
    """

    X = np.column_stack((Fr, beta, Cd))
    y = var

    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=feature_names)  # Update with actual names

    # prepare data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #create and train the model
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    #make prediction
    y_pred = model.predict(X)

    # plot comparison of alpha between random forest regression and SRH-2D
    plot_alpha_diff_random_forest_regression_vs_truth(y_pred, y, Fr, beta, Cd)

    # plot alpha_diff histogram
    plot_alpha_diff_distributions_simple_model(y)

    # Use SHAP to explain predictions
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Convert SHAP values and feature data to a DataFrame for easier plotting
    shap_values_df = pd.DataFrame(shap_values, columns=X.columns)
    mean_shap_values = np.abs(shap_values_df).mean().sort_values(ascending=False)  # Sort by importance

    # Plot a custom summary bar plot for SHAP values
    bPlot_alpha_diff_dependence_SHAP_bar_SRH_2D=True
    if bPlot_alpha_diff_dependence_SHAP_bar_SRH_2D:
        plt.figure(figsize=(6, 4))
        plt.barh(mean_shap_values.index, mean_shap_values.values, color="deepskyblue")
        plt.xlabel(r"SHAP value (impact on flow partition error $\alpha_{error}$)", fontsize=18)
        plt.ylabel("Variables", fontsize=18)
        #plt.title("Customized SHAP Summary Plot", fontsize=16, fontweight="bold")
        plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature on top
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid(axis="x", linestyle="--", alpha=0.7)

        plt.xlim([0, 0.035])

        # Add text for each bar
        for index, value in enumerate(mean_shap_values):
            plt.text(value, index, f'{value:.3f}', fontsize=18,
                     va='center')  # `value` sets the x-position, `index` sets the y-position

        plt.savefig("alpha_diff_dependence_SHAP_bar_SRH_2D.png", dpi=300, bbox_inches='tight', pad_inches=0)

        plt.show()

    bPlot_alpha_diff_dependence_SHAP_summary_SRH_2D=True
    if bPlot_alpha_diff_dependence_SHAP_summary_SRH_2D:
        fig, ax = plt.subplots(figsize=(8, 6))

        shap.summary_plot(shap_values, X, feature_names=feature_names, cmap="coolwarm", show=False)

        # Customizing the plot
        # Access the current figure and axis after the plot is created
        fig = plt.gcf()  # Get the current figure
        ax = plt.gca()  # Get the current axis

        # plt.title("Customized SHAP Summary Plot", fontsize=18, fontweight='bold')
        ax.set_xlabel(r"SHAP value (impact on the flow partition error $\alpha_{error}$)", fontsize=16)
        ax.set_ylabel("Variables", fontsize=16)

        # Customize tick labels
        ax.tick_params(axis='x', labelsize=16)
        # ax.tick_params(axis='y', labelsize=12)

        # Add grid lines (optional)
        # plt.grid(True, linestyle='--', alpha=0.5)

        # Show the customized plot
        plt.tight_layout()

        fig.savefig("alpha_diff_dependence_SHAP_summary_SRH_2D.png", dpi=300, bbox_inches='tight', pad_inches=0)

        plt.show()

def feature_dependence_alpha_SRH_2D():
    """
    Analyze feature depdendence of alpha on Fr, beta, and Cd
        1. perform random forest regressor on Fr, beta, and Cd
        2. analyze using SHAP package

    Returns
    -------

    """

    #read data: the csv file should have been created before calling this function.
    data = pd.read_csv('combined_results_simple_SRH_2D.csv')

    print(data.head())

    Fr = data['Fr'].to_numpy()
    beta = data['beta'].to_numpy()
    Cd = data['Cd'].to_numpy()
    alpha_SRH_2D = data['alpha_SRH_2D'].to_numpy()

    feature_names = [r"$Fr$", r"$\beta$", r"$C_d$"]

    # analyze using SHAP (global analysis)
    using_SHAP(Fr, beta, Cd, alpha_SRH_2D, feature_names)

    # analyze using SHAP (local analysis)
    using_SHAP_local(Fr, beta, Cd, alpha_SRH_2D, feature_names)

def feature_dependence_alpha_diff_simple_model_with_SRH_2D():
    """
    diff_alpha=f(Fr, beta, Cd). This function uses ML method to analyze the dependance
    and their relative importance.

    Returns
    -------

    """

    #read data: the csv file should have been created before calling this function.
    data = pd.read_csv('combined_results_simple_SRH_2D.csv')

    #print(data.head())

    Fr = data['Fr'].to_numpy()
    beta = data['beta'].to_numpy()
    Cd = data['Cd'].to_numpy()
    alpha_simple = data['alpha_simple'].to_numpy()
    alpha_SRH_2D = data['alpha_SRH_2D'].to_numpy()
    alpha_diff = abs(alpha_simple - alpha_SRH_2D)

    feature_names = [r"$Fr$", r"$\beta$", r"$C_d$"]

    # analyze using SHAP
    using_SHAP_for_alpha_diff(Fr, beta, Cd, alpha_diff, feature_names)

def compare_simple_model_with_SRH_2D():
    """
    Compare the simple model solution with SRH-2D results.

    :return:
    """

    # load the simple model data
    with np.load('Fr_beta_C_d_h2prime_h2_alpha_arrays_simple.npz') as data:
        iCases = data['iCases']
        Frs = data['Frs']
        betas = data['betas']
        C_ds = data['C_ds']
        alpha_results = data['alpha']
        bConverged_results = data['bConverged']

    # load the SRH-2D data
    with np.load('Fr_beta_C_d_h2prime_h2_alpha_arrays_SRH_2D.npz') as data:
        iCases_SRH_2D = data['iCases']
        Frs_SRH_2D = data['Frs']
        betas_SRH_2D = data['betas']
        Cds_SRH_2D = data['Cds']
        alpha_results_SRH_2D = data['alpha']
        bConverged_results_SRH_2D = data['bSuccess']

    nTotal = Fr_n * beta_n * C_d_n

    if nTotal!=(Frs.shape[0]*betas.shape[0]*C_ds.shape[0]):
        print("nTotal and the dimensions of simple model results are not consistent.")
        exit()

    if nTotal!=(Frs_SRH_2D.shape[0]*Frs_SRH_2D.shape[1]*Frs_SRH_2D.shape[2]):
        print("nTotal and the dimensions of SRH-2D results are not consistent.")
        exit()

    #make the data to be in 1D arrays
    Frs_1D = np.zeros(nTotal)
    betas_1D = np.zeros(nTotal)
    Cds_1D = np.zeros(nTotal)
    iCases_1D = np.zeros(nTotal)

    alphas_1D         = np.zeros(nTotal)
    bConverged_1D      = np.zeros(nTotal)
    iCases_1D_SRH_2D = np.zeros(nTotal)
    alphas_1D_SRH_2D = np.zeros(nTotal)
    bConverged_1D_SRH_2D = np.zeros(nTotal)

    # combined results: iCase, Fr, beta, Cd, alpha_simple, alpha_SRH_2D, bConverged_simple, bConverged_SRH_2D
    combined_results_simple_SRH_2D = np.zeros((nTotal, 8))

    iCase = 0
    for iFr in range(Fr_n):
        for ibeta in range(beta_n):
            for iCd in range(C_d_n):
                Frs_1D[iCase] = Frs[iFr]
                betas_1D[iCase] = betas[ibeta]
                Cds_1D[iCase] = C_ds[iCd]
                iCases_1D[iCase] = iCase

                alphas_1D[iCase] = alpha_results[iFr,ibeta,iCd]
                bConverged_1D[iCase] = bConverged_results[iFr,ibeta,iCd]

                iCases_1D_SRH_2D[iCase] = iCases_SRH_2D[iFr, ibeta, iCd]
                alphas_1D_SRH_2D[iCase] = alpha_results_SRH_2D[iFr, ibeta, iCd]
                bConverged_1D_SRH_2D[iCase] = bConverged_results_SRH_2D[iFr, ibeta, iCd]

                combined_results_simple_SRH_2D[iCase, 0] = iCase
                combined_results_simple_SRH_2D[iCase, 1] = Frs_1D[iCase]
                combined_results_simple_SRH_2D[iCase, 2] = betas_1D[iCase]
                combined_results_simple_SRH_2D[iCase, 3] = Cds_1D[iCase]
                combined_results_simple_SRH_2D[iCase, 4] = alphas_1D[iCase]
                combined_results_simple_SRH_2D[iCase, 5] = alphas_1D_SRH_2D[iCase]
                combined_results_simple_SRH_2D[iCase, 6] = bConverged_1D[iCase]
                combined_results_simple_SRH_2D[iCase, 7] = bConverged_1D_SRH_2D[iCase]

                iCase += 1

    np.savetxt("combined_results_simple_SRH_2D.csv", combined_results_simple_SRH_2D, delimiter=",",
               fmt="%.2f",
               header="iCase,Fr,beta,Cd,alpha_simple,alpha_SRH_2D,bConverged_simple,bConverged_SRH_2D",
               comments="")

    # define convergence criteria
    simple_model_converged_criterion = bConverged_1D > 0.5
    simple_model_diverged_criterion = bConverged_1D < 0.5
    SRH_2D_model_converged_criterion = bConverged_1D_SRH_2D > 0.5
    SRH_2D_model_diverged_criterion = bConverged_1D_SRH_2D < 0.5
    converged_criterion_both = simple_model_converged_criterion & SRH_2D_model_converged_criterion
    diverged_criterion_both = simple_model_diverged_criterion | SRH_2D_model_diverged_criterion

    print(
        f"simple_model_converged_criterion: {np.sum(simple_model_converged_criterion)} out of {len(simple_model_converged_criterion)}")
    print(
        f"SRH_2D_model_converged_criterion: {np.sum(SRH_2D_model_converged_criterion)} out of {len(SRH_2D_model_converged_criterion)}")
    print(f"converged_criterion_both: {np.sum(converged_criterion_both)} out of {len(converged_criterion_both)}")

    # define convergence/divergence criterion
    # converged_criterion=simple_model_converged_criterion
    # diverged_criterion = simple_model_diverged_criterion
    # converged_criterion = SRH_2D_model_converged_criterion
    # diverged_criterion = SRH_2D_model_diverged_criterion
    converged_criterion = converged_criterion_both
    diverged_criterion = diverged_criterion_both

    alphas_1D_converged = alphas_1D[converged_criterion]
    alphas_1D_diverged = alphas_1D[diverged_criterion]
    alphas_1D_SRH_2D_converged = alphas_1D_SRH_2D[converged_criterion]
    alphas_1D_SRH_2D_diverged = alphas_1D_SRH_2D[diverged_criterion]
    Frs_1D_converged = Frs_1D[converged_criterion]
    Frs_1D_diverged = Frs_1D[diverged_criterion]
    betas_1D_converged = betas_1D[converged_criterion]
    betas_1D_diverged = betas_1D[diverged_criterion]
    Cds_1D_converged = Cds_1D[converged_criterion]
    Cds_1D_diverged = Cds_1D[diverged_criterion]

    iCases_1D_converged = iCases_1D_SRH_2D[converged_criterion]
    iCases_1D_diverged = iCases_1D_SRH_2D[diverged_criterion]

    # Set the print options to display the full array
    np.set_printoptions(threshold=np.inf)

    converged_parameters = np.column_stack(
        (iCases_1D_converged, Frs_1D_converged, betas_1D_converged, Cds_1D_converged))
    diverged_parameters = np.column_stack((iCases_1D_diverged, Frs_1D_diverged, betas_1D_diverged, Cds_1D_diverged))

    # print("converged_parameters: iCase, Fr, beta, Cd\n", converged_parameters)
    print("diverged_parameters: iCase, Fr, beta, Cd\n", diverged_parameters)

    #plot the disbtributions of alpha values from both simple model and SRH-2D
    plot_alpha_distributions_simple_model_vs_SRH_2D(alphas_1D_converged, alphas_1D_SRH_2D_converged)

    color_with_vars = ["none", "Fr", "beta", "Cd"]

    for color_with_var in color_with_vars:

        #make plot
        fig, ax = plt.subplots(figsize=(6, 6))

        # Define a colormap (e.g., 'coolwarm' colormap)
        cmap = plt.get_cmap('coolwarm')


        # Plot the diagonal line
        ax.plot([0, 1], [0, 1], color='black', linestyle='--')

        if color_with_var=="none":
            color_var = None
        elif color_with_var=="Fr":
            color_var = Frs_1D_converged
        elif color_with_var=="beta":
            color_var = betas_1D_converged
        elif color_with_var=="Cd":
            color_var = Cds_1D_converged

        rmse = np.sqrt(np.mean((alphas_1D_converged - alphas_1D_SRH_2D_converged) ** 2))
        r2 = r2_score(alphas_1D_converged, alphas_1D_SRH_2D_converged)

        if color_var is None:
            scatter_plot = ax.scatter(alphas_1D_converged, alphas_1D_SRH_2D_converged,
                                      facecolors='none', edgecolor='black',
                                      marker='o', s=60)  #s=60
            #ax.text(0.05, 0.8, f"RMSE = {rmse:.3f}", fontsize=14, color="black", ha="left")
            #ax.text(0.05, 0.75, f"$r^2$ = {r2:.3f}", fontsize=14, color="black", ha="left")
        elif color_with_var=="Fr" or color_with_var=="beta":
            scatter_plot = ax.scatter(alphas_1D_converged, alphas_1D_SRH_2D_converged, c=color_var, edgecolors='none',
                                      cmap=cmap, vmin=0, vmax=1, marker='o', s=60)  # s=60
        elif color_with_var=="Cd":
            scatter_plot = ax.scatter(alphas_1D_converged, alphas_1D_SRH_2D_converged, c=color_var, edgecolors='none',
                                      cmap=cmap, vmin=0, vmax=80, marker='o', s=60)  # s=60

        ax.text(0.05, 0.8, f"RMSE = {rmse:.3f}", fontsize=16, color="black", ha="left")
        ax.text(0.05, 0.75, f"$r^2$ = {r2:.3f}", fontsize=16, color="black", ha="left")

        if color_var is not None:
            cbar = plt.colorbar(scatter_plot, shrink=1.0, fraction=0.03, pad=0.04, aspect=40)

        if color_with_var == "Fr":
            cbar.set_label(r'$Fr$', fontsize=18)
        elif color_with_var == "beta":
            cbar.set_label(r'$\beta$', fontsize=18)
        elif color_with_var == "Cd":
            cbar.set_label(r'$C_d$', fontsize=18)

        if color_var is not None:
            #cbar.set_label(r'$C_d$', fontsize=14)
            #cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])  # Set the desired tick locations

            cbar.ax.tick_params(labelsize=16)  # Set tick font size

            # Additional customizations if needed
            #cbar.outline.set_linewidth(0.5)  # Adjust colorbar outline width

        # set the limit for the x and y axes
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        ax.set_aspect('equal')

        # set x and y axes label and font size
        ax.set_xlabel(r'$\alpha$ from simple model', fontsize=18)
        ax.set_ylabel(r'$\alpha$ from SRH-2D', fontsize=18)

        # show the ticks on both axes and set the font size
        ax.tick_params(axis='both', which='major', labelsize=16)
        # xtick_spacing=1
        # ax.xaxis.set_major_locator(tick.MultipleLocator(xtick_spacing))
        #ax.xaxis.set_major_formatter(tick.FormatStrFormatter('%.1f'))

        # show legend, set its location, font size, and turn off the frame
        # plt.legend(loc='lower right', fontsize=14, frameon=False)

        fig.savefig("alpha_comparison_simple_vs_SRH_2D_"+color_with_var+".png", dpi=300, bbox_inches='tight', pad_inches=0)

        plt.show()

def plot_alpha_results():
    """
    plot the alpha results (using SRH-2D simulation data)
    Only plot the result for a few selected Fr values. Then, alpha is a function of beta and Cd.

    Returns
    -------

    """

    #read data: the csv file should have been created before calling this function.
    data = pd.read_csv('combined_results_simple_SRH_2D.csv')

    print(data.head())

    Fr = data['Fr'].to_numpy()
    beta = data['beta'].to_numpy()
    Cd = data['Cd'].to_numpy()
    alpha_SRH_2D = data['alpha_SRH_2D'].to_numpy()

    Frs_to_show = [0.05, 0.35, 0.65, 0.95]

    for iFrs in range(len(Frs_to_show)):
        Fr_to_show = Frs_to_show[iFrs]

        print("Ploting Fr = ", Fr_to_show)

        Fr_selected = []
        beta_selected = []
        Cd_selected = []
        alpha_SRH_2D_selected = []

        for i in range(len(Fr)):
            if abs(Fr[i] - Fr_to_show) < 0.001:
                Fr_selected.append(Fr[i])
                beta_selected.append(beta[i])
                Cd_selected.append(Cd[i])
                alpha_SRH_2D_selected.append(alpha_SRH_2D[i])

        Fr_selected = np.array(Fr_selected)
        beta_selected = np.array(beta_selected)
        Cd_selected = np.array(Cd_selected)
        alpha_SRH_2D_selected = np.array(alpha_SRH_2D_selected)

        print(alpha_SRH_2D_selected)

        # Create the scatter plot
        fig, ax = plt.subplots(figsize=(6, 6))

        #scatter_plot = ax.scatter(Fr_selected, beta_selected, s=beta_selected*100, c=alpha_SRH_2D_selected, vmin=0, vmax=1, cmap='coolwarm')  #, alpha=0.7
        scatter_plot = ax.scatter(beta_selected, Cd_selected, c=alpha_SRH_2D_selected, vmin=0,
                                  vmax=1, cmap='coolwarm')  # , alpha=0.7

        cbar = plt.colorbar(scatter_plot, shrink=1.0, fraction=0.03, pad=0.04, aspect=40)
        cbar.set_label(r"$\alpha$", fontsize=18)
        cbar.ax.tick_params(labelsize=16)  # Set colorbar label number font size

        # Create a manual size legend
        #for size in [5, 30, 50, 80, 100]:  # Specify sizes you want to display in the legend
        #    ax.scatter([], [], c='gray', alpha=0.5, s=size, label=f'{size/100:.2f}')

        #ax.legend(scatterpoints=1, frameon=True, labelspacing=1, title=r"$\beta$")

        # set the limit for the x and y axes
        ax.set_xlim([0, 1])
        ax.set_ylim([-2, 82])

        # set x and y axes label and font size
        ax.set_xlabel(r'$\beta$', fontsize=18)
        ax.set_ylabel(r'$C_d$', fontsize=18)

        # show the ticks on both axes and set the font size
        ax.tick_params(axis='both', which='major', labelsize=16)

        ax.set_title(r"$Fr$ = " + f'{Fr_to_show:.2f}', fontsize=18)

        fig.savefig("alpha_result_SRH_2D_" + str(iFrs).zfill(4) +".png", dpi=300, bbox_inches='tight', pad_inches=0)

        plt.show()

def solve_for_flume_experiment_cases():
    """
    Solve for the four flume experiments.
    Returns
    -------

    """
    case_IDs = [1, 2, 3, 4]
    # case_IDs = [1]

    Frs = [0.076, 0.097, 0.126, 0.163]
    Cds = [64.8, 53.4, 45.5, 40]
    betas = [0.5, 0.5, 0.5, 0.5]
    alphas_exp = [0.868, 0.857, 0.846, 0.836]

    alphas_simple = [0.0, 0.0, 0.0, 0.0]

    for idx, case_ID in enumerate(case_IDs):
        print("Solving the flume experiment case: ", case_ID)

        solution, bChoked, bConverged = solve_LWD_for_given_Fr_beta_C_d(Frs[idx], betas[idx], Cds[idx], True)

        # Store the results in the data cubes
        if not bChoked:
            alphas_simple[idx] = solution[2]
        else:
            alphas_simple[idx] = solution[3]

    print("alphas_exp = ", alphas_exp)
    print("alphas_simple = ", alphas_simple)

    experiments_results_vs_simple_solutions = {"case_IDs":case_IDs,
                                               "Frs": Frs,
                                               "Cds":Cds,
                                               "betas":betas,
                                               "alphas_exp":alphas_exp,
                                               "alphas_simple":alphas_simple
                                               }
    with open("experiments_results_vs_simple_solutions.json", mode="w") as file:
        json.dump(experiments_results_vs_simple_solutions, file, indent=4, separators=(",", ": "))


def plot_contour_from_vtk(case_ID, Fr, beta, Cd, alpha_SRH_2D, alpha_simple, rect_x, rect_y, rect_width, rect_height, filename):
    #load data from vtk file: water depht and velocity

    if not os.path.exists(filename):
        return None

    # Load the VTK file
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(filename)  # Replace with your VTK file path
    reader.ReadAllScalarsOn()  # Ensure all scalar fields are read
    reader.ReadAllVectorsOn()  # Ensure all vector fields are read
    reader.Update()

    # Get the unstructured grid data
    data = reader.GetOutput()

    #print("data = ", data)

    # Extract cell centers and scalar data
    cell_centers = vtk.vtkCellCenters()
    cell_centers.SetInputData(data)
    cell_centers.Update()

    # Get points and scalar values from the cell centers
    points = np.array(
        [cell_centers.GetOutput().GetPoint(i)[:2] for i in range(cell_centers.GetOutput().GetNumberOfPoints())])

    # Flip x coordinates
    points[:, 0] = -points[:, 0]

    # Extract values from cell data (assuming the scalar field is at cell centers)
    Velocity_m_p_s = data.GetCellData().GetArray("Velocity_m_p_s")
    Vel_Mag_m_p_s = data.GetCellData().GetArray("Vel_Mag_m_p_s")
    Water_Depth_m = data.GetCellData().GetArray("Water_Depth_m")

    if Velocity_m_p_s is None:
        raise ValueError("No Velocity_m_p_s data found at cell centers. Please check your VTK file.")

    # Convert scalar data to a numpy array
    Vel_mag_np = np.array([Vel_Mag_m_p_s.GetTuple1(i) for i in range(Vel_Mag_m_p_s.GetNumberOfTuples())])

    # Check if points and scalars have compatible shapes
    if len(points) != len(Vel_mag_np):
        raise ValueError("Mismatch between number of cell centers and scalar values.")

    # Create a grid for contour plotting
    x = points[:, 0]
    y = points[:, 1]
    z = Vel_mag_np

    # Create a regular grid interpolated from the scattered data
    xi = np.linspace(x.min()+0.01, x.max()-0.01, 420)
    yi = np.linspace(y.min()+0.01, y.max()-0.01, 60)
    X, Y = np.meshgrid(xi, yi)

    # Interpolate the scalar field onto the grid
    Z = griddata(points, z, (X, Y), method="linear")

    vmin = Z.min()
    vmax = Z.max()

    # Plot the contour
    fig, ax = plt.subplots(figsize=(42, 6))

    contour = ax.contourf(X, Y, Z, levels=20, cmap="coolwarm", vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(contour) #, shrink=1.0, fraction=0.1, pad=0.04, aspect=40)

    cbar.set_label(label="Velocity (m/s)", fontsize=48)

    tick_positions = np.linspace(vmin, vmax, 5)

    cbar.set_ticks(tick_positions)  # Apply custom ticks
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    cbar.ax.tick_params(labelsize=60)  # Set tick font size

    rectangle = plt.Rectangle((rect_x, rect_y), rect_width, rect_height, linewidth=2, edgecolor="white", facecolor="none")
    ax.add_patch(rectangle)

    #ax.set_xlim(-50.1+0.02, 25.1-0.02)
    #ax.set_ylim(0+0.02, 10-0.02)
    ax.set_xlim(xi.min(), xi.max())
    ax.set_ylim(yi.min(), yi.max())

    ax.set_xlabel("x (m)", fontsize=70)
    ax.set_ylabel("y (m)", fontsize=70)

    ax.tick_params(axis='x', labelsize=60)
    ax.tick_params(axis='y', labelsize=60)

    ax.set_title(rf"$Fr$ = {Fr:.2f}, $\beta$ = {beta:.2f}, $C_d$ = {Cd:.2f}, $\alpha_{{SRH-2D}}$ = {alpha_SRH_2D:.2f}, $\alpha_{{simple}}$ = {alpha_simple:.2f}", fontsize=65)

    # Show the customized plot
    plt.tight_layout()

    fig.savefig("vel_mag_contour_"+ str(case_ID).zfill(4) +".png", dpi=300, bbox_inches='tight', pad_inches=0)

    plt.show()

def plot_vel_contour_four_example_cases():
    """
    Plot the velocity contour using four examples. The results are from SRH-2D simulations.

    Returns
    -------

    """

    case_IDs = [1, 13, 642, 772]
    #case_IDs = [1]

    Frs = [0.05, 0.05, 0.65, 0.75]
    Cds = [8.89, 26.67, 17.78, 17.78]
    betas = [0.05, 0.15, 0.45, 0.75]
    alphas = [0.11, 0.4, 0.71, 0.90]

    # read data: the csv file should have been created before calling this function.
    data = pd.read_csv('combined_results_simple_SRH_2D.csv')

    print(data.head())

    Fr = data['Fr'].to_numpy()
    beta = data['beta'].to_numpy()
    Cd = data['Cd'].to_numpy()
    alpha_simple = data['alpha_simple'].to_numpy()
    alpha_SRH_2D = data['alpha_SRH_2D'].to_numpy()

    rect_x, rect_y = -0.1, 0  # Bottom-left corner of the rectangle
    rect_width =  0.2 # Width of the rectangle
    rect_heights = [] # Height of the rectangle
    for beta_SRH_2D in betas:
        rect_heights.append(10.0*(1-beta_SRH_2D))

    for i, case_ID in enumerate(case_IDs):
        print("plotting case_ID = ", case_ID)
        print("    Fr = ", Fr[case_ID])
        print("    beta = ", beta[case_ID])
        print("    Cd = ", Cd[case_ID])
        print("    alpha_simple = ", alpha_simple[case_ID])
        print("    alpha_SRH_2D = ", alpha_SRH_2D[case_ID])

        vtkFileName = "results/case_result_"+ str(case_ID).zfill(4) +".vtk"
        plot_contour_from_vtk(case_ID, Frs[i], betas[i], Cds[i], alphas[i], alpha_simple[case_ID], rect_x, rect_y, rect_width, rect_heights[i], vtkFileName)

if __name__ == "__main__":

    #solve the simple model for a porous structure (LWD) in open channel flows
    solve_LWD()

    #Test: example solve: solve_LWD_for_given_Fr_beta_C_d(Fr, beta, C_d, bLWDMomentumChange)
    #solve_LWD_for_given_Fr_beta_C_d(0.2, 0.5, 10, True)
    #solve_LWD_for_given_Fr_beta_C_d(0.45, 0.55, 26.7, True)

    #Solve for the four flume experiment cases
    solve_for_flume_experiment_cases()

    #plot result alpha = f(Fr,beta,Cd)
    plot_alpha_results()

    # compare simple model with SRH-2D
    compare_simple_model_with_SRH_2D()

    # analyze the feature dependencde of alpha on Fr, beta, and Cd
    # alpha = f(Fr,beta,Cd)
    feature_dependence_alpha_SRH_2D()

    # analyze the feature dependence of difference between simple model and SRH-2D results
    # diff_alpha = f(Fr,beta,Cd)
    feature_dependence_alpha_diff_simple_model_with_SRH_2D()

    # plot velocity contour of four example cases
    plot_vel_contour_four_example_cases()

    print("All done!")
