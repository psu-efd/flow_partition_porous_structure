#Analytical solution for open channel hydraulics with a porous large woody debris (LWD)
#The solution is for the flow split (percentage of flow goes through the opening and LWD).

#Note: This version of the code uses the old the formulation which assumes the upstream water depth does not change if 
# the flow is not choked. If choked, flow backup upstream. This approach used the traditional open channel flow through 
# a contaction. But it does not work well because there is alwasy backwater effect. Solution is not stable when the Fr is high,
# and/or contraction is large. 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.optimize import least_squares
from scipy.optimize import curve_fit

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model

from sklearn.ensemble import RandomForestRegressor
#from sklearn.inspection import plot_partial_dependence
import shap

from gplearn.genetic import SymbolicRegressor

from customized_SHAP_waterfall import custom_waterfall

import os

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
C_A_min = 0.0
C_A_max = 80.0
C_A_n = 10

# Define the system of nonlinear equations for non-choking condition
def equations_not_choking(vars, Fr, beta, C_A, bLWDMomentumChange):
    h2prime, h2, alpha = vars

    eq1 = h2 + 1.0/2.0 * alpha**2/beta**2 / h2**2 * Fr**2 - (1 + 1.0/2.0*Fr**2)         # equation 1
    eq2 = h2prime**2 - h2**2 - C_A/h2 * ( (1-alpha)/(1-beta) )**2 * Fr**2                    # equation 2

    if bLWDMomentumChange: #if include the momentum change through LWD
        eq2 = eq2 + 2/h2prime* ((1-alpha)/(1-beta) )**2 * Fr**2 * (1-h2prime/h2)

    eq3 = h2prime + 1.0/2.0 * ((1-alpha)/(1-beta) )**2 * Fr**2 /h2prime**2 - (1 + 1.0/2.0*Fr**2)     # equation 3

    #print("residuals = ", eq1,eq2,eq3)

    return [eq1, eq2, eq3]

def jacobian_not_choking(vars, Fr, beta, C_A, bLWDMomentumChange):
    """
    Jacobian of equations_not_choking
    Parameters
    ----------
    vars
    Fr
    beta
    C_A
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
    jacobian[1,1] = -2*h2 +C_A/h2**2 *(1-alpha)**2/(1-beta)**2*Fr**2
    jacobian[1,2] = 2*C_A/h2*(1-alpha)/(1-beta)**2*Fr**2

    if bLWDMomentumChange:
        jacobian[1,0] += 2*(1-alpha)**2/(1-beta)**2*Fr**2*(1-h2prime/h2)
        jacobian[1,1] += 2*(1-alpha)**2/(1-beta)**2*Fr**2*h2prime**2/h2**2
        jacobian[1,2] -= 4/h2prime*(1-alpha)/(1-beta)**2*Fr**2*(1-h2prime/h2)

    jacobian[2,0] = 1 - (1-alpha)**2/(1-beta)**2*Fr**2/h2prime**3
    jacobian[2,1] = 0.0
    jacobian[2,2] = -(1-alpha)/(1-beta)**2*Fr**2/h2prime**2

    return jacobian

# Define the system of nonlinear equations for choking condition
def equations_choking(vars, Fr, beta, C_A, bLWDMomentumChange):
    h1star, h2prime, h2, alpha = vars

    eq1 = h1star + 1.0/2.0 * Fr**2 / h1star**2 - 3.0/2.0*(alpha/beta*Fr)**(2.0/3.0)  #equation 1

    eq2 = h1star + 1.0/2.0 * Fr**2 / h1star**2 - h2prime - 1.0/2.0 * (1-alpha)**2/(1-beta)**2 * Fr**2 / h2prime**2   # equation 2

    eq3 = h2prime**2 - h2**2 - C_A/h2 * ( (1-alpha)/(1-beta) )**2 * Fr**2                    # equation 3

    if bLWDMomentumChange: #if include the momentum change through LWD
        eq3 = eq3 + 2/h2prime* ((1-alpha)/(1-beta) )**2 * Fr**2 * (1-h2prime/h2)

    eq4 = h2prime -(alpha*Fr/beta)**(2.0/3.0)     # equation 4

    #print("residuals = ", eq1,eq2,eq3,eq4)

    return [eq1, eq2, eq3, eq4]

def jacobian_choking(vars, Fr, beta, C_A, bLWDMomentumChange):
    """
    Jacobian of equations_choking
    Parameters
    ----------
    vars
    Fr
    beta
    C_A
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
    jacobian[2,2] = -2*h2 +C_A/h2**2 *(1-alpha)**2/(1-beta)**2*Fr**2
    jacobian[2,3] = 2*C_A/h2*(1-alpha)/(1-beta)**2*Fr**2

    if bLWDMomentumChange:
        jacobian[2,1] += 2*(1-alpha)**2/(1-beta)**2*Fr**2*(1-h2prime/h2)
        jacobian[2,2] += 2*(1-alpha)**2/(1-beta)**2*Fr**2*h2prime**2/h2**2
        jacobian[2,3] -= 4/h2prime*(1-alpha)/(1-beta)**2*Fr**2*(1-h2prime/h2)

    jacobian[3,0] = 0
    jacobian[3,1] = 0
    jacobian[3,2] = 1
    jacobian[3,3] = 2.0/3.0/alpha**(1.0/3.0)*(Fr/beta)**(2.0/3.0)

    return jacobian


def solve_LWD_for_given_Fr_beta_C_A(Fr, beta, C_A, bLWDMomentumChange):
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

    print("Fr, beta, C_A, bLWDMomentumChange=", Fr, beta, C_A, bLWDMomentumChange)

    # if solver_option==1:      #solve with fsolve
    #try to solve with fsolve first.
    print("    Solving with fsolve: assuming not choked condition...")

    #try with the not-choking condition first to see whether it works. In not-choking condition,
    #there are three unknows and three equations
    # unknowns: h2prime, h2, alpha
    initial_guess = [1.0, 1.0, 0.5]

    #solution, infodict, ier, mesg = fsolve(equations_not_choking, initial_guess, args=(Fr,beta,C_A,bLWDMomentumChange), full_output=True)
    solution, infodict, ier, mesg = fsolve(equations_not_choking, initial_guess, args=(Fr, beta, C_A, bLWDMomentumChange), fprime=jacobian_not_choking, full_output=True)

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
    # print("residuals=",np.isclose(equations_not_choking(solution,Fr,beta,C_A,bLWDMomentumChange),[0,0,0]))
    #print("residuals=", equations_not_choking(solution, Fr, beta, C_A, bLWDMomentumChange))

    # check positivity of solution
    if solution[0] < 0.0 or solution[1] < 0.0 or solution[2] < 0.0:
        bConverged = 0

    #if the solution is not converged, assume the flow is choked (no solution)
    if bConverged==0:
        print("        The flow may be choked. Solve with the choking condition.")

        # unknowns: h1star, h2prime, h2, alpha
        initial_guess = [1.5, 1.1, 0.8, 0.8]

        #solution, infodict, ier, mesg = fsolve(equations_choking, initial_guess, args=(Fr, beta, C_A, bLWDMomentumChange), full_output=True)
        solution, infodict, ier, mesg = fsolve(equations_choking, initial_guess, args=(Fr, beta, C_A, bLWDMomentumChange), fprime=jacobian_choking, full_output=True)

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
        #print("    residuals=",np.isclose(equations_choking(solution,Fr,beta,C_A,bLWDMomentumChange),[0,0,0,0]))
        #print("    residuals=", equations_choking(solution, Fr, beta, C_A, bLWDMomentumChange))

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
                               args=(Fr,beta,C_A,bLWDMomentumChange),
                               jac=jacobian_not_choking,
                               method='dogbox')
        # Extract the solution
        solution = result.x

        #print("residuals=",equations_not_choking(solution, Fr, beta, C_A, bLWDMomentumChange))
        #print("residuals=", np.isclose(equations_not_choking(solution, Fr, beta, C_A, bLWDMomentumChange), [0, 0, 0]))

        residuals = np.isclose(equations_not_choking(solution, Fr, beta, C_A, bLWDMomentumChange), [0, 0, 0])

        # Check if the optimization was successful
        if result.success:
            bConverged = 1
            # Display the solution
            print("Fr, beta, C_A, bLWDMomentumChange=", Fr, beta, C_A, bLWDMomentumChange)
            print(f"Solution: h2prime = {solution[0]}, h2 = {solution[1]}, alpha = {solution[2]}")
        else:
            bConverged = 0
            print("Optimization failed:", result.message)
            print("Fr, beta, C_A, bLWDMomentumChange=", Fr, beta, C_A, bLWDMomentumChange)
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
                                   args=(Fr, beta, C_A, bLWDMomentumChange),
                                   jac=jacobian_choking, method='dogbox')
            # Extract the solution
            solution = result.x

            # Check if the optimization was successful
            if result.success:
                bConverged = 1
                bChoked = 1

                # Display the solution
                print("    Fr, beta, C_A, bLWDMomentumChange=", Fr, beta, C_A, bLWDMomentumChange)
                print(f"    Solution: h1star= {solution[0]},h2prime = {solution[1]}, h2 = {solution[2]}, alpha = {solution[3]}")
            else:
                bConverged = 0
                bChoked = 0
                print("    Optimization failed:", result.message)
                print("    Fr, beta, C_A, bLWDMomentumChange=", Fr, beta, C_A, bLWDMomentumChange)
                print("    result = ", result)

            # check positivity of solution
            if solution[0] < 0.0 or solution[1] < 0.0 or solution[2] < 0.0 or solution[3] < 0.0 or solution[3] > 1.0:
                bConverged = 0

            print("    residuals=", equations_choking(solution, Fr, beta, C_A, bLWDMomentumChange))
            print("    residuals=",
                  np.isclose(equations_choking(solution, Fr, beta, C_A, bLWDMomentumChange), [0, 0, 0, 0]))

            residuals = np.isclose(equations_choking(solution, Fr, beta, C_A, bLWDMomentumChange), [0, 0, 0, 0])

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
    Solve LWD in open channel flow

    Returns
    -------

    """

    #define samples on parameter space
    #upstream Froude number
    Frs = np.linspace(Fr_min, Fr_max, Fr_n)
    #opening width fraction at LWD
    betas = np.linspace(beta_min, beta_max, beta_n)
    #LWD dimensionless drag parameter
    C_As = np.linspace(C_A_min, C_A_max, C_A_n)

    #whether to consider the momentum change within LWD
    bLWDMomentumChange = True

    # Initialize an empty 3D data cubes to store results
    iCases_results = np.empty((len(Frs), len(betas), len(C_As)))
    h2prime_results = np.empty((len(Frs), len(betas), len(C_As)))
    h2_results = np.empty((len(Frs), len(betas), len(C_As)))
    alpha_results = np.empty((len(Frs), len(betas), len(C_As)))
    bChoked_results = np.empty((len(Frs), len(betas), len(C_As)))
    bConverged_results = np.empty((len(Frs), len(betas), len(C_As)))

    # Loop over each combination of Fr, beta, C_A values
    iCase = 0
    for i in range(len(Frs)):  # Loop over Fr-values
        for j in range(len(betas)):  # Loop over beta-values
            for k in range(len(C_As)):  # Loop over C_A-values
                # Extract the current values of Fr, beta, C_A
                Fr = Frs[i]
                beta = betas[j]
                C_A = C_As[k]

                # Perform some computation using Fr, beta, and C_A
                solution, bChoked, bConverged = solve_LWD_for_given_Fr_beta_C_A(Fr, beta, C_A, bLWDMomentumChange)

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
    print("Saving Fr_beta_C_A_h2prime_h2_alpha_arrays_simple.npz")
    np.savez_compressed('Fr_beta_C_A_h2prime_h2_alpha_arrays_simple.npz',
                        iCases=iCases_results,
                        Frs=Frs, betas=betas, C_As=C_As,
                        h2prime=h2prime_results, h2=h2_results, alpha=alpha_results,
                        bChoked=bChoked_results, bConverged=bConverged_results)

def compare_simple_model_with_srhFoam():
    """
    Compare the solution of simple model with srhFoam results.

    :return:
    """

    # load the simple model data
    with np.load('Fr_beta_C_A_h2prime_h2_alpha_arrays_simple.npz') as data:
        iCases = data['iCases']
        Frs = data['Frs']
        betas = data['betas']
        C_As = data['C_As']
        alpha_results = data['alpha']
        bConverged_results = data['bConverged']

    # load the srhFoam data
    with np.load('Fr_beta_C_A_h2prime_h2_alpha_arrays_srhFoam_no_porosity.npz') as data:
    #with np.load('Fr_beta_C_A_h2prime_h2_alpha_arrays_srhFoam_with_porosity.npz') as data:
        iCases_srhFoam = data['iCases']
        Frs_srhFoam = data['Frs']
        betas_srhFoam = data['betas']
        Cds_srhFoam = data['Cds']
        alpha_results_srhFoam = data['alpha']
        bConverged_results_srhFoam = data['bSuccess']

    nTotal = Fr_n * beta_n * C_A_n

    if nTotal!=(Frs.shape[0]*betas.shape[0]*C_As.shape[0]):
        print("nTotal and the dimensions of simple model results are not consistent.")
        exit()

    if nTotal!=(Frs_srhFoam.shape[0]*Frs_srhFoam.shape[1]*Frs_srhFoam.shape[2]):
        print("nTotal and the dimensions of srhFoam results are not consistent.")
        exit()

    #make the data to be in 1D arrays
    Frs_1D = np.zeros(nTotal)
    betas_1D = np.zeros(nTotal)
    Cds_1D = np.zeros(nTotal)
    iCases_1D = np.zeros(nTotal)

    alphas_1D         = np.zeros(nTotal)
    bConverged_1D      = np.zeros(nTotal)
    iCases_1D_srhFoam = np.zeros(nTotal)
    alphas_1D_srhFoam = np.zeros(nTotal)
    bConverged_1D_srhFoam = np.zeros(nTotal)

    iCase = 0
    for iFr in range(Fr_n):
        for ibeta in range(beta_n):
            for iCd in range(C_A_n):
                Frs_1D[iCase] = Frs[iFr]
                betas_1D[iCase] = betas[ibeta]
                Cds_1D[iCase] = C_As[iCd]
                iCases_1D[iCase] = iCase

                alphas_1D[iCase] = alpha_results[iFr,ibeta,iCd]
                bConverged_1D[iCase] = bConverged_results[iFr,ibeta,iCd]

                iCases_1D_srhFoam[iCase] = iCases_srhFoam[iFr, ibeta, iCd]
                alphas_1D_srhFoam[iCase] = alpha_results_srhFoam[iFr, ibeta, iCd]
                bConverged_1D_srhFoam[iCase] = bConverged_results_srhFoam[iFr, ibeta, iCd]

                iCase += 1

    #make plot
    fig, ax = plt.subplots(figsize=(6, 6))

    # Define a colormap (e.g., 'coolwarm' colormap)
    cmap = plt.get_cmap('coolwarm')

    color_with_var = "Fr"
    color_with_var = "beta"
    color_with_var = "Cd"

    #define criteria
    simple_model_converged_criterion = bConverged_1D > 0.5
    simple_model_diverged_criterion = bConverged_1D < 0.5
    srhFoam_model_converged_criterion = bConverged_1D_srhFoam > 0.5
    srhFoam_model_diverged_criterion = bConverged_1D_srhFoam < 0.5
    converged_criterion_both = simple_model_converged_criterion & srhFoam_model_converged_criterion
    diverged_criterion_both = simple_model_diverged_criterion | srhFoam_model_diverged_criterion

    print(f"simple_model_converged_criterion: {np.sum(simple_model_converged_criterion)} out of {len(simple_model_converged_criterion)}")
    print(f"srhFoam_model_converged_criterion: {np.sum(srhFoam_model_converged_criterion)} out of {len(srhFoam_model_converged_criterion)}")
    print(f"converged_criterion_both: {np.sum(converged_criterion_both)} out of {len(converged_criterion_both)}")

    #define convergence/divergence criterion
    converged_criterion=simple_model_converged_criterion
    diverged_criterion = simple_model_diverged_criterion
    #converged_criterion = srhFoam_model_converged_criterion
    #diverged_criterion = srhFoam_model_diverged_criterion
    #converged_criterion = converged_criterion_both
    #diverged_criterion = diverged_criterion_both

    alphas_1D_converged = alphas_1D[converged_criterion]
    alphas_1D_diverged = alphas_1D[diverged_criterion]
    alphas_1D_srhFoam_converged = alphas_1D_srhFoam[converged_criterion]
    alphas_1D_srhFoam_diverged = alphas_1D_srhFoam[diverged_criterion]
    Frs_1D_converged = Frs_1D[converged_criterion]
    Frs_1D_diverged = Frs_1D[diverged_criterion]
    betas_1D_converged = betas_1D[converged_criterion]
    betas_1D_diverged = betas_1D[diverged_criterion]
    Cds_1D_converged = Cds_1D[converged_criterion]
    Cds_1D_diverged = Cds_1D[diverged_criterion]

    iCases_1D_converged = iCases_1D_srhFoam[converged_criterion]
    iCases_1D_diverged = iCases_1D_srhFoam[diverged_criterion]

    # Set the print options to display the full array
    np.set_printoptions(threshold=np.inf)

    converged_parameters = np.column_stack((iCases_1D_converged, Frs_1D_converged,betas_1D_converged,Cds_1D_converged))
    diverged_parameters  = np.column_stack((iCases_1D_diverged, Frs_1D_diverged, betas_1D_diverged, Cds_1D_diverged))

    #print("converged_parameters: iCase, Fr, beta, Cd\n", converged_parameters)
    print("diverged_parameters: iCase, Fr, beta, Cd\n", diverged_parameters)

    # Plot the diagonal line
    ax.plot([0, 1], [0, 1], color='black', linestyle='--')

    scatter_plot = ax.scatter(alphas_1D_converged, alphas_1D_srhFoam_converged, c=Cds_1D_converged, edgecolors='none', cmap=cmap, marker='o', s=60)  #s=60
    #scatter_plot = ax.scatter(alphas_1D, alphas_1D_srhFoam, c=betas_1D, edgecolors='none', cmap=cmap, marker='o', s=60, alpha=1-bConverged_1D_srhFoam)
    #scatter_plot = ax.scatter(alphas_1D, alphas_1D_srhFoam, c=Frs_1D, edgecolors='none', cmap=cmap, marker='o',
    #                          s=60)  # alpha=Frs_1D
    #scatter_plot = ax.scatter(alphas_1D, alphas_1D_srhFoam, c=betas_1D, edgecolors='none', cmap=cmap, marker='o',
    #                          s=Cds_1D)  # alpha=Frs_1D

    cbar = plt.colorbar(scatter_plot, fraction=0.03,pad=0.04,aspect=40)
    cbar.set_label(r'$\beta$', fontsize=14)
    #cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])  # Set the desired tick locations

    cbar.ax.tick_params(labelsize=12)  # Set tick font size

    # Additional customizations if needed
    cbar.outline.set_linewidth(0.5)  # Adjust colorbar outline width


    # set the limit for the x and y axes
    #ax.set_xlim([0, 1])
    #ax.set_ylim([0, 1])

    ax.set_aspect('equal')

    # set x and y axes label and font size
    ax.set_xlabel(r'$\alpha$ from simple model', fontsize=16)
    ax.set_ylabel(r'$\alpha$ from \textit{srhFoam}', fontsize=16)

    # show the ticks on both axes and set the font size
    ax.tick_params(axis='both', which='major', labelsize=12)
    # xtick_spacing=1
    # ax.xaxis.set_major_locator(tick.MultipleLocator(xtick_spacing))
    #ax.xaxis.set_major_formatter(tick.FormatStrFormatter('%.1f'))

    # show legend, set its location, font size, and turn off the frame
    # plt.legend(loc='lower right', fontsize=14, frameon=False)
    plt.show()

    fig.savefig("alpha_comparison_simple_vs_srhFoam.png", dpi=300, bbox_inches='tight', pad_inches=0)

def compare_simple_model_with_SRH_2D():
    """
    Compare the simple model solution with SRH-2D results.

    :return:
    """

    # load the simple model data
    with np.load('Fr_beta_C_A_h2prime_h2_alpha_arrays_simple.npz') as data:
        iCases = data['iCases']
        Frs = data['Frs']
        betas = data['betas']
        C_As = data['C_As']
        alpha_results = data['alpha']
        bConverged_results = data['bConverged']

    # load the SRH-2D data
    with np.load('Fr_beta_C_A_h2prime_h2_alpha_arrays_SRH_2D_Windows.npz') as data:
        iCases_SRH_2D = data['iCases']
        Frs_SRH_2D = data['Frs']
        betas_SRH_2D = data['betas']
        Cds_SRH_2D = data['Cds']
        alpha_results_SRH_2D = data['alpha']
        bConverged_results_SRH_2D = data['bSuccess']

    nTotal = Fr_n * beta_n * C_A_n

    if nTotal!=(Frs.shape[0]*betas.shape[0]*C_As.shape[0]):
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
            for iCd in range(C_A_n):
                Frs_1D[iCase] = Frs[iFr]
                betas_1D[iCase] = betas[ibeta]
                Cds_1D[iCase] = C_As[iCd]
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

    exit()

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

        if color_var is None:
            scatter_plot = ax.scatter(alphas_1D_converged, alphas_1D_SRH_2D_converged, facecolors='none', edgecolor='black', alpha=0.5, marker='o', s=60)  #s=60
            ax.text(0.2, 0.8, f"RMSE = {rmse:.3f}" , fontsize=14, color="black", ha="center")
        else:
            scatter_plot = ax.scatter(alphas_1D_converged, alphas_1D_SRH_2D_converged, c=color_var, edgecolors='none',
                                      cmap=cmap, marker='o', s=60)  # s=60

        if color_var is not None:
            cbar = plt.colorbar(scatter_plot, fraction=0.03,pad=0.04,aspect=40)

        if color_with_var == "Fr":
            cbar.set_label(r'$Fr$', fontsize=14)
        elif color_with_var == "beta":
            cbar.set_label(r'$\beta$', fontsize=14)
        elif color_with_var == "Cd":
            cbar.set_label(r'$C_d$', fontsize=14)

        if color_var is not None:
            #cbar.set_label(r'$C_d$', fontsize=14)
            #cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])  # Set the desired tick locations

            cbar.ax.tick_params(labelsize=12)  # Set tick font size

            # Additional customizations if needed
            cbar.outline.set_linewidth(0.5)  # Adjust colorbar outline width

        # set the limit for the x and y axes
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        ax.set_aspect('equal')

        # set x and y axes label and font size
        ax.set_xlabel(r'$\alpha$ from simple model', fontsize=16)
        ax.set_ylabel(r'$\alpha$ from SRH-2D', fontsize=16)

        # show the ticks on both axes and set the font size
        ax.tick_params(axis='both', which='major', labelsize=12)
        # xtick_spacing=1
        # ax.xaxis.set_major_locator(tick.MultipleLocator(xtick_spacing))
        #ax.xaxis.set_major_formatter(tick.FormatStrFormatter('%.1f'))

        # show legend, set its location, font size, and turn off the frame
        # plt.legend(loc='lower right', fontsize=14, frameon=False)
        #plt.show()

        fig.savefig("alpha_comparison_simple_vs_SRH_2D_"+color_with_var+".png", dpi=300, bbox_inches='tight', pad_inches=0)

        plt.close()

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
                     fontsize=12,
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
    ax.set_xlabel(r'Flow partition $\alpha$', fontsize=16)
    ax.set_ylabel('Count', fontsize=16)
    #ax.set_ylim(0, 110)

    # Set font size of axis ticks
    ax.tick_params(axis='both', which='major', labelsize=12)  # Adjust labelsize as needed

    # Adjusting layout for better spacing
    # plt.tight_layout()

    ax.legend(
        loc="upper left",          # Position of the legend
        fontsize=12,                # Font size of the legend text
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
                     fontsize=12,
                     horizontalalignment='left',
                     verticalalignment='top'
                     )

    # Adding labels and title
    ax.set_xlabel(r'$\alpha_{error}$ due to the simple model', fontsize=16)
    ax.set_ylabel('Count', fontsize=16)
    #ax.set_ylim(0, 110)

    # Set font size of axis ticks
    ax.tick_params(axis='both', which='major', labelsize=12)  # Adjust label size as needed

    # Adjusting layout for better spacing
    # plt.tight_layout()

    #ax.legend(
    #    loc="upper left",          # Position of the legend
    #    fontsize=12,                # Font size of the legend text
    #    frameon=True,               # Add a frame around the legend
    #    fancybox=False,              # Rounded edges on the legend box
    #    shadow=False,                # Add a shadow to the legend box
        #title="Legend Title",       # Title for the legend
        #title_fontsize=12           # Font size for the legend title
    #)

    figureFileName = "alpha_diff_distributions_simple_model.png"
    fig.savefig(figureFileName, dpi=300, bbox_inches='tight', pad_inches=0.1)

    # Display the figure
    plt.show()

def plot_alpha_random_forest_regression_vs_truth(alpha_random_forest, alpha_truth):
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

    rmse = np.sqrt(np.mean((alpha_random_forest - alpha_truth) ** 2))

    scatter_plot = ax.scatter(alpha_random_forest, alpha_truth, facecolors='none', edgecolor='black',
                                  alpha=0.5, marker='o', s=60)  # s=60
    ax.text(0.2, 0.8, f"RMSE = {rmse:.3f}", fontsize=14, color="black", ha="center")

    #cbar = plt.colorbar(scatter_plot, fraction=0.03, pad=0.04, aspect=40)

    # set the limit for the x and y axes
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    ax.set_aspect('equal')

    # set x and y axes label and font size
    ax.set_xlabel(r'$\alpha$ from random forest regression', fontsize=16)
    ax.set_ylabel(r'$\alpha$ from SRH-2D', fontsize=16)

    # show the ticks on both axes and set the font size
    ax.tick_params(axis='both', which='major', labelsize=14)
    # xtick_spacing=1
    # ax.xaxis.set_major_locator(tick.MultipleLocator(xtick_spacing))
    # ax.xaxis.set_major_formatter(tick.FormatStrFormatter('%.1f'))

    # show legend, set its location, font size, and turn off the frame
    # plt.legend(loc='lower right', fontsize=14, frameon=False)

    fig.savefig("alpha_comparison_random_forest_vs_SRH_2D.png", dpi=300, bbox_inches='tight',
                pad_inches=0)

    plt.show()

    #plt.close()

def plot_alpha_diff_random_forest_regression_vs_truth(alpha_diff_random_forest, alpha_diff_truth):
    """
    Plot and compare alpha diff values from random forest regression model and truth
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
    ax.plot([0, 0.5], [0, 0.5], color='black', linestyle='--')

    color_var = None

    rmse = np.sqrt(np.mean((alpha_diff_random_forest - alpha_diff_truth) ** 2))

    scatter_plot = ax.scatter(alpha_diff_random_forest, alpha_diff_truth, facecolors='none', edgecolor='black',
                                  alpha=0.5, marker='o', s=60)  # s=60
    ax.text(0.2, 0.4, f"RMSE = {rmse:.3f}", fontsize=14, color="black", ha="center")

    #cbar = plt.colorbar(scatter_plot, fraction=0.03, pad=0.04, aspect=40)

    # set the limit for the x and y axes
    #ax.set_xlim([0, 1])
    #ax.set_ylim([0, 1])

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

    fig.savefig("alpha_diff_comparison_random_forest_vs_truth.png", dpi=300, bbox_inches='tight',
                pad_inches=0)

    plt.show()

    #plt.close()

def using_random_forest(Fr, beta, Cd, var, feature_names):
    """
    UUsing a Random Forest Regressor for Feature Importance

    :param Fr:
    :param beta:
    :param Cd:
    :param var:
    :return:
    """

    X = np.column_stack((Fr, beta, Cd))
    y = var

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # R2 score using the model's score method
    r2_score = model.score(X_test, y_test)
    print("RandomForestRegressor R2 Score:", r2_score)

    # Get feature importances
    importances = model.feature_importances_

    print("importances=",importances)

    # Plot feature importances
    plt.bar(feature_names, importances)
    plt.ylabel("Feature Importance")
    #plt.title("Random Forest Feature Importance for alpha_diff=f(Fr, beta, Cd)")
    plt.show()

def symbolic_regression():
    """
        Symbolic regression (does not work well).

        :param Fr:
        :param beta:
        :param Cd:
        :param alpha_diff:
        :return:
        """

    # read data: the csv file should have been created before calling this function.
    data = pd.read_csv('combined_results_simple_SRH_2D.csv')

    print(data.head())

    Fr = data['Fr'].to_numpy()
    beta = data['beta'].to_numpy()
    Cd = data['Cd'].to_numpy()
    alpha_SRH_2D = data['alpha_SRH_2D'].to_numpy()

    feature_names = [r"$Fr$", r"$\beta$", r"$C_d$"]

    # Prepare the input matrix for symbolic regression
    X = np.column_stack((Fr, beta, Cd))  # Shape should be (n_samples, n_features)

    # Initialize the Symbolic Regressor
    est_gp = SymbolicRegressor(population_size=5000,
                               generations=100,
                               stopping_criteria=0.01,
                               p_crossover=0.7,
                               p_subtree_mutation=0.1,
                               p_hoist_mutation=0.05,
                               p_point_mutation=0.1,
                               max_samples=0.9,
                               verbose=1,
                               parsimony_coefficient=0.01,
                               random_state=0)

    # Fit the model to the data
    est_gp.fit(X, alpha_SRH_2D)

    # Print the best found symbolic expression
    print("Best symbolic expression:", est_gp._program)

    # Predict alpha using the symbolic regression model
    alpha_pred = est_gp.predict(X)

    # Calculate RMSE for model performance evaluation
    rmse = np.sqrt(np.mean((alpha_SRH_2D - alpha_pred) ** 2))
    print(f"RMSE of the symbolic regression model: {rmse}")

    # Scatter plot to compare true vs predicted values
    plt.figure(figsize=(8, 8))
    plt.scatter(alpha_SRH_2D, alpha_pred, color="blue", label="Predicted vs True")
    plt.plot([0, 1], [0, 1], color="red", linestyle="--")
    plt.xlabel("True Alpha")
    plt.ylabel("Predicted Alpha")
    plt.title("True vs Predicted Alpha (Symbolic Regression)")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.show()

def nonlinear_regression():
    """
    Nonlinear regression.

    :param Fr:
    :param beta:
    :param Cd:
    :param alpha_diff:
    :return:
    """

    # read data: the csv file should have been created before calling this function.
    data = pd.read_csv('combined_results_simple_SRH_2D.csv')

    print(data.head())

    Fr = data['Fr'].to_numpy()
    beta = data['beta'].to_numpy()
    Cd = data['Cd'].to_numpy()
    alpha_SRH_2D = data['alpha_SRH_2D'].to_numpy()

    feature_names = [r"$Fr$", r"$\beta$", r"$C_d$"]

    # Define the regression function based on the proposed formula
    def regression_model(X, c0, c1, c2, c3, c4, c5, c6):
        Fr, beta, Cd = X
        #return c0 + c1 * Fr + c2 * np.sqrt(beta) + c3 * np.sqrt(Cd)  + c4 * np.sqrt(Cd) * np.sqrt(beta)
        #return c0 + c1 * Fr + c2 * np.sqrt(beta) + c3 * np.sqrt(Cd) + c4 * np.sqrt(Cd) / np.sqrt(beta)
        #return c0 + c1 * Fr + c2 * np.sqrt(beta) + c3 * np.log(Cd+1) + c4 * np.log(Cd+1) * np.sqrt(beta)
        #return c0 + c1 * Fr + c2 * np.power(beta, c5) + c3 * np.log(Cd/80 + 1) + c4 * Fr * np.log(Cd/80 + 1) # * np.sqrt(beta)
        return c0 + c1 * Fr + c2 * np.power(beta, c5) + c3 * np.power(Cd / 80, c6) + c4 * Fr * np.power(Cd/80, c6)  * np.power(beta, c5)
        #return c0 + c1 * beta + c2 * np.log(Cd+1)

    # Perform curve fitting
    # `p0` is an optional initial guess for the parameters
    popt, pcov = curve_fit(regression_model, (Fr, beta, Cd), alpha_SRH_2D, p0=[0, 1, 1, 1, 1, 1, 1])

    # Extract fitted parameters
    c0, c1, c2, c3, c4, c5, c6 = popt
    print(f"Fitted coefficients:\nc0 = {c0}\nc1 = {c1}\nc2 = {c2}\nc3 = {c3}\nc4 = {c4}\nc5 = {c5}")

    # Predict alpha values using the fitted model
    alpha_pred = regression_model((Fr, beta, Cd), c0, c1, c2, c3, c4, c5, c6)

    # Calculate the root mean square error (RMSE) for the fitted model
    rmse = np.sqrt(np.mean((alpha_SRH_2D - alpha_pred) ** 2))
    print(f"RMSE of the fitted model: {rmse}")

    # Plot the true vs predicted values of alpha
    plt.figure(figsize=(8, 8))
    plt.scatter(alpha_SRH_2D, alpha_pred, c=Cd, label="Predicted vs True")
    plt.colorbar(label="Color Scale (Variable)")
    plt.plot([0, 1], [0, 1], color="black", linestyle="--")

    # Set labels, aspect ratio, and legend
    plt.xlabel(r"$\alpha$ from SRH-2D")
    plt.ylabel(r"$\alpha$ from regression")
    #plt.title("True vs Predicted Alpha")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.gca().set_aspect('equal', adjustable='box')  # Set aspect ratio to be equal
    plt.legend()

    plt.show()

def polynomial_regression():
    """
    Polynomial regression.

    :return:
    """

    # read data: the csv file should have been created before calling this function.
    data = pd.read_csv('combined_results_simple_SRH_2D.csv')

    print(data.head())

    Fr = data['Fr'].to_numpy()
    beta = data['beta'].to_numpy()
    Cd = data['Cd'].to_numpy()
    alpha_SRH_2D = data['alpha_SRH_2D'].to_numpy()

    Cd = Cd/80   #scale Cd

    # Stack the independent variables into a single matrix
    X = np.column_stack((Fr, beta, Cd))

    # Define the polynomial degree and fit the model
    degree = 2
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X)

    # Fit the initial polynomial regression model
    model = LinearRegression()
    model.fit(X_poly, alpha_SRH_2D)

    # Retrieve feature names and coefficients
    feature_names = poly.get_feature_names_out(["Fr", "beta", "Cd"])
    coefficients = model.coef_

    # Set a threshold for coefficient importance (e.g., drop coefficients close to zero)
    threshold = 0.02
    significant_indices = [i for i, coef in enumerate(coefficients) if abs(coef) > threshold]
    significant_terms = [(feature_names[i], coefficients[i]) for i in significant_indices]

    # Print the simplified polynomial equation for the initial model
    initial_simplified_equation = f"{model.intercept_:.3f} "
    for name, coef in significant_terms:
        initial_simplified_equation += f"+ ({coef:.3f}) * {name} "

    print("Simplified Polynomial Equation after Applying Threshold on Initial Model:")
    print(initial_simplified_equation)

    # Create a reduced dataset with only the significant terms
    X_reduced = X_poly[:, significant_indices]
    reduced_model = LinearRegression()
    reduced_model.fit(X_reduced, alpha_SRH_2D)

    # Print the polynomial equation of the reduced_model
    reduced_intercept = reduced_model.intercept_
    reduced_coefficients = reduced_model.coef_

    reduced_equation = f"{reduced_intercept:.3f} "
    for (name, coef) in zip([feature_names[i] for i in significant_indices], reduced_coefficients):
        reduced_equation += f"+ ({coef:.3f}) * {name} "

    print("\nPolynomial Equation for Reduced Model:")
    print(reduced_equation)

    # Make predictions with the reduced model
    alpha_pred = reduced_model.predict(X_reduced)

    # Calculate and print RMSE for evaluation
    rmse = np.sqrt(mean_squared_error(alpha_SRH_2D, alpha_pred))
    r2 = r2_score(alpha_SRH_2D, alpha_pred)

    print(f"Root Mean Squared Error: {rmse}")
    print(f"R-squared: {r2}")

    # Plot the true vs predicted values of alpha
    # make plot
    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot the diagonal line
    ax.plot([0, 1], [0, 1], color='black', linestyle='--')

    #scatter_plot = ax.scatter(alpha_pred, alpha_SRH_2D, facecolors='none', edgecolor='black', alpha=0.5, marker='o', s=60)  # s=60
    scatter_plot = ax.scatter(alpha_pred, alpha_SRH_2D, c=Cd, cmap='coolwarm', alpha=0.5, marker='o', s=60)  # s=60
    ax.text(0.05, 0.8, f"RMSE = {rmse:.3f}", fontsize=14, color="black", ha="left")
    ax.text(0.05, 0.75, f"$r^2$ = {r2:.3f}", fontsize=14, color="black", ha="left")

    cbar = plt.colorbar(scatter_plot, shrink=1.0, fraction=0.03, pad=0.04, aspect=40)
    # Customize colorbar
    cbar.set_label(r"$C_d$", fontsize=16)  # Set colorbar title font size
    cbar.ax.tick_params(labelsize=14)  # Set colorbar label number font size

    # set the limit for the x and y axes
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    ax.set_aspect('equal')

    # set x and y axes label and font size
    ax.set_xlabel(r'$\alpha$ from polynomial regression', fontsize=16)
    ax.set_ylabel(r'$\alpha$ from SRH-2D', fontsize=16)

    # show the ticks on both axes and set the font size
    ax.tick_params(axis='both', which='major', labelsize=14)
    # xtick_spacing=1
    # ax.xaxis.set_major_locator(tick.MultipleLocator(xtick_spacing))
    # ax.xaxis.set_major_formatter(tick.FormatStrFormatter('%.1f'))

    # show legend, set its location, font size, and turn off the frame
    # plt.legend(loc='lower right', fontsize=14, frameon=False)

    fig.savefig("alpha_comparison_polynomial_vs_SRH_2D.png", dpi=300, bbox_inches='tight',
                pad_inches=0)

    plt.show()

    # plt.close()

def polynomial_regression_simplify():
    """
    After calling the polynomial_regression(), here we do some further simplification.

    :return:
    """

    # read data: the csv file should have been created before calling this function.
    data = pd.read_csv('combined_results_simple_SRH_2D.csv')

    print(data.head())

    Fr = data['Fr'].to_numpy()
    beta = data['beta'].to_numpy()
    Cd = data['Cd'].to_numpy()
    alpha_SRH_2D = data['alpha_SRH_2D'].to_numpy()

    Cd = Cd/80   #scale Cd

    # Polynomial     Equation for Reduced Model:
    #    -0.086 + (-0.039) * Fr + (1.939) * beta + (0.686) * Cd + (0.036) * Fr
    #    beta + (-0.993) * beta ^ 2 + (-0.237) * beta Cd + (-0.364) * Cd ^ 2

    # Make predictions with the reduced model
    alpha_pred = np.zeros(len(alpha_SRH_2D))
    for i in range (0, len(alpha_SRH_2D)):
        #alpha_pred[i] = (-0.086 -0.039*Fr[i] + 1.939*beta[i] + 0.686*Cd[i] + 0.036*Fr[i]*beta[i] -0.993*beta[i]**2 -0.237*beta[i]*Cd[i] -0.364*Cd[i]**2)
        alpha_pred[i] = (-0.086 - 0.0 * Fr[i] + 1.939 * beta[i] + 0.686 * Cd[i] + 0.0 * Fr[i] * beta[i] - 0.993 * beta[i] ** 2 - 0.237 * beta[i] * Cd[i] - 0.364 * Cd[i] ** 2)

    # Calculate and print RMSE for evaluation
    rmse = np.sqrt(mean_squared_error(alpha_SRH_2D, alpha_pred))
    r2 = r2_score(alpha_SRH_2D, alpha_pred)

    print(f"Root Mean Squared Error: {rmse}")
    print(f"R-squared: {r2}")

    # Plot the true vs predicted values of alpha
    # make plot
    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot the diagonal line
    ax.plot([0, 1], [0, 1], color='black', linestyle='--')

    #scatter_plot = ax.scatter(alpha_pred, alpha_SRH_2D, facecolors='none', edgecolor='black', alpha=0.5, marker='o', s=60)  # s=60
    scatter_plot = ax.scatter(alpha_pred, alpha_SRH_2D, c=Cd, cmap='coolwarm', alpha=0.5, marker='o', s=60)  # s=60
    ax.text(0.05, 0.8, f"RMSE = {rmse:.3f}", fontsize=14, color="black", ha="left")
    ax.text(0.05, 0.75, f"$r^2$ = {r2:.3f}", fontsize=14, color="black", ha="left")

    cbar = plt.colorbar(scatter_plot, shrink=1.0, fraction=0.03, pad=0.04, aspect=40)
    # Customize colorbar
    cbar.set_label(r"$C_d$", fontsize=16)  # Set colorbar title font size
    cbar.ax.tick_params(labelsize=14)  # Set colorbar label number font size

    # set the limit for the x and y axes
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    ax.set_aspect('equal')

    # set x and y axes label and font size
    ax.set_xlabel(r'$\alpha$ from polynomial regression', fontsize=16)
    ax.set_ylabel(r'$\alpha$ from SRH-2D', fontsize=16)

    # show the ticks on both axes and set the font size
    ax.tick_params(axis='both', which='major', labelsize=14)
    # xtick_spacing=1
    # ax.xaxis.set_major_locator(tick.MultipleLocator(xtick_spacing))
    # ax.xaxis.set_major_formatter(tick.FormatStrFormatter('%.1f'))

    # show legend, set its location, font size, and turn off the frame
    # plt.legend(loc='lower right', fontsize=14, frameon=False)

    fig.savefig("alpha_comparison_simplifed_polynomial_vs_SRH_2D.png", dpi=300, bbox_inches='tight',
                pad_inches=0)

    plt.show()

    # plt.close()



def using_SHAP(Fr, beta, Cd, var, feature_names):
    """
    Using SHAP (SHapley Additive exPlanations) for Feature Importance. This is global analysis across the whole dataset.
    Random Forest Regressor is still used.

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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # R2 score using the model's score method
    r2_score = model.score(X_test, y_test)
    print("R2 Score:", r2_score)

    y_pred = model.predict(X_test)

    #plot comparison of alpha between random forest regression and SRH-2D
    #plot_alpha_random_forest_regression_vs_truth(y_pred, y_test)

    #exit()

    # Use SHAP to explain predictions
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)


    # Convert SHAP values and feature data to a DataFrame for easier plotting
    shap_values_df = pd.DataFrame(shap_values, columns=X.columns)
    mean_shap_values = np.abs(shap_values_df).mean().sort_values(ascending=False)  # Sort by importance

    # Dependence plot
    if 1:
        shap_values_temp = explainer(X)

        # Optionally, color the plot by another feature
        #shap.dependence_plot(r"$Fr$", shap_values_temp.values, X, interaction_index=r"$\beta$") # Color by a feature
        #shap.dependence_plot(r"$\beta$", shap_values_temp.values, X, interaction_index=r"$C_d$")  # Color by a feature
        #shap.dependence_plot(r"$C_d$", shap_values_temp.values, X, interaction_index=r"$\beta$")  # Color by a feature

        # Define features to plot
        features_to_plot = [r"$Fr$", r"$\beta$", r"$C_d$"]  # Specify the features you want to plot
        features_to_color = [r"$\beta$", r"$C_d$", r"$\beta$"]  # Specify the features you want to plot
        #features_to_color = [None, None, None]  # Specify the features you want to plot

        # Determine the y-axis range by finding the min and max SHAP values across all features
        y_min = np.min(shap_values_temp.values)
        y_max = np.max(shap_values_temp.values)

        # Set up subplots
        fig, axes = plt.subplots(1, len(features_to_plot), figsize=(5 * len(features_to_plot), 4), sharey=True)

        # Plot each feature in a separate subplot
        for i, feature  in enumerate(features_to_plot):
            ax = axes[i]
            shap.dependence_plot(feature, shap_values_temp.values, X, interaction_index=features_to_color[i], ax=ax, show=False)
            ax.set_ylim(y_min, y_max)  # Set y-axis limits to be the same across plots

        # Display the plot
        plt.tight_layout()

        plt.savefig("alpha_dependence_plot_SHAP_SRH_2D.png", dpi=300, bbox_inches='tight', pad_inches=0)

        plt.show()

        exit()

    if 1:
        alpha = var

        # Define features to plot
        feature_names_to_plot = [r"$Fr$", r"$\beta$", r"$C_d$"]  # Specify the features you want to plot
        feature_names_to_color = [r"$\beta$", r"$C_d$", r"$\beta$"]  # Specify the features you want to plot
        #features_to_color = [None, None, None]  # Specify the features you want to plot

        features_to_plot = [Fr, beta, Cd]

        # Determine the y-axis range by finding the min and max SHAP values across all features
        alpha_min = np.min(alpha)
        alpha_max = np.max(alpha)

        # Set up subplots
        fig, axes = plt.subplots(1, len(features_to_plot), figsize=(5 * len(features_to_plot), 4), sharey=True)

        # Plot each feature in a separate subplot
        for i, feature in enumerate(features_to_plot):
            ax = axes[i]
            ax.scatter(feature, alpha)
            ax.set_xlabel(feature_names_to_plot[i], fontsize=14)
            ax.set_ylabel(r"$\alpha$", fontsize=14)
            ax.set_ylim(0, 1)  # Set y-axis limits to be the same across plots

        # Display the plot
        plt.tight_layout()

        plt.savefig("alpha_dependence_plot_SRH_2D.png", dpi=300, bbox_inches='tight', pad_inches=0)

        plt.show()

        exit()

    # Plot a custom summary bar plot for SHAP values
    if 0:
        plt.figure(figsize=(6, 4))
        plt.barh(mean_shap_values.index, mean_shap_values.values, color="deepskyblue")

        plt.xlabel(r"SHAP value (impact on flow partition $\alpha$)", fontsize=14)
        plt.ylabel("Variables", fontsize=14)
        #plt.title("Customized SHAP Summary Plot", fontsize=16, fontweight="bold")
        plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature on top
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(axis="x", linestyle="--", alpha=0.7)

        # Add text for each bar
        for index, value in enumerate(mean_shap_values):
            plt.text(value, index, f'{value:.3f}', va='center')  # `value` sets the x-position, `index` sets the y-position

        plt.savefig("alpha_dependence_SHAP_bar_SRH_2D.png", dpi=300, bbox_inches='tight', pad_inches=0)

        plt.show()

        exit()

    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)

    # Customizing the plot
    # Access the current figure and axis after the plot is created
    fig = plt.gcf()  # Get the current figure
    ax = plt.gca()  # Get the current axis

    # plt.title("Customized SHAP Summary Plot", fontsize=18, fontweight='bold')
    ax.set_xlabel(r"SHAP value (impact on flow partition $\alpha$)", fontsize=14)
    ax.set_ylabel("Variables", fontsize=14)

    # Customize the color bar legend
    cbar = plt.gcf().axes[-1]  # Access the last axis, which is the color bar in SHAP plot
    cbar.set_ylabel("Variable value", fontsize=12)  # Change "Feature value" to "Variable value"

    # Customize tick labels
    ax.tick_params(axis='x', labelsize=12)
    # ax.tick_params(axis='y', labelsize=12)

    # Add grid lines (optional)
    # plt.grid(True, linestyle='--', alpha=0.5)

    # Show the customized plot
    # plt.tight_layout()

    fig.savefig("alpha_dependence_SHAP_summary_SRH_2D.png", dpi=300, bbox_inches='tight', pad_inches=0)

    # plt.show()

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

def using_SHAP_for_alpha_dff(Fr, beta, Cd, var, feature_names):
    """
    Using SHAP (SHapley Additive exPlanations) for Feature Importance. Random Forest Regressor
     is still used.

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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # R2 score using the model's score method
    r2_score = model.score(X_test, y_test)
    print("R2 Score:", r2_score)

    y_pred = model.predict(X_test)

    # plot comparison of alpha between random forest regression and SRH-2D
    plot_alpha_diff_random_forest_regression_vs_truth(y_pred, y_test)

    # plot alpha_diff histogram
    plot_alpha_diff_distributions_simple_model(y)

    #exit()

    # Use SHAP to explain predictions
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Convert SHAP values and feature data to a DataFrame for easier plotting
    shap_values_df = pd.DataFrame(shap_values, columns=X.columns)
    mean_shap_values = np.abs(shap_values_df).mean().sort_values(ascending=False)  # Sort by importance

    # Plot a custom summary bar plot for SHAP values
    if 0:
        plt.figure(figsize=(6, 4))
        plt.barh(mean_shap_values.index, mean_shap_values.values, color="deepskyblue")
        plt.xlabel(r"SHAP value (impact on flow partition error $\alpha_{error}$)", fontsize=14)
        plt.ylabel("Variables", fontsize=14)
        #plt.title("Customized SHAP Summary Plot", fontsize=16, fontweight="bold")
        plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature on top
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(axis="x", linestyle="--", alpha=0.7)

        # Add text for each bar
        for index, value in enumerate(mean_shap_values):
            plt.text(value, index, f'{value:.3f}',
                     va='center')  # `value` sets the x-position, `index` sets the y-position

        plt.savefig("alpha_diff_dependence_SHAP_bar_SRH_2D.png", dpi=300, bbox_inches='tight', pad_inches=0)

        plt.show()

        exit()

    #exit()

    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)

    # Customizing the plot
    # Access the current figure and axis after the plot is created
    fig = plt.gcf()  # Get the current figure
    ax = plt.gca()  # Get the current axis

    # plt.title("Customized SHAP Summary Plot", fontsize=18, fontweight='bold')
    ax.set_xlabel(r"SHAP value (impact on the flow partition error $\alpha_{error}$)", fontsize=14)
    ax.set_ylabel("Variables", fontsize=14)

    # Customize the color bar legend
    cbar = plt.gcf().axes[-1]  # Access the last axis, which is the color bar in SHAP plot
    cbar.set_ylabel("Variable value", fontsize=12)  # Change "Feature value" to "Variable value"

    # Customize tick labels
    ax.tick_params(axis='x', labelsize=12)
    # ax.tick_params(axis='y', labelsize=12)

    # Add grid lines (optional)
    # plt.grid(True, linestyle='--', alpha=0.5)

    # Show the customized plot
    # plt.tight_layout()

    fig.savefig("alpha_diff_dependence_SHAP_summary_SRH_2D.png", dpi=300, bbox_inches='tight', pad_inches=0)

    # plt.show()


def using_PDP(Fr, beta, Cd, var, feature_names):
    """
    Using Partial Dependence Plots for Feature Importance. Random Forest Regressor
     is still used.

    :param Fr:
    :param beta:
    :param Cd:
    :param alpha_diff:
    :return:
    """

    X = np.column_stack((Fr, beta, Cd))
    y = var

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # R2 score using the model's score method
    r2_score = model.score(X_test, y_test)
    print("R2 Score:", r2_score)

    # Plot partial dependence plots
    plot_partial_dependence(model, X, features=[0, 1, 2], feature_names=feature_names)
    plt.show()

def feature_dependence_alpha_SRH_2D():
    """

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

    #analyze using random forest
    #using_random_forest(Fr, beta, Cd, alpha_SRH_2D, feature_names)

    # analyze using SHAP
    using_SHAP(Fr, beta, Cd, alpha_SRH_2D, feature_names)

    #using_SHAP_local(Fr, beta, Cd, alpha_SRH_2D, feature_names)

    # analyze using PDP
    #using_PDP(Fr, beta, Cd, alpha_SRH_2D, feature_names)



def feature_dependence_alpha_diff_simple_model_with_SRH_2D():
    """
    diff_alpha=f(Fr, beta, Cd). This function uses ML method to analyze the dependance and their relative importance.

    Returns
    -------

    """


    #read data: the csv file should have been created before calling this function.
    data = pd.read_csv('combined_results_simple_SRH_2D.csv')

    print(data.head())

    Fr = data['Fr'].to_numpy()
    beta = data['beta'].to_numpy()
    Cd = data['Cd'].to_numpy()
    alpha_simple = data['alpha_simple'].to_numpy()
    alpha_SRH_2D = data['alpha_SRH_2D'].to_numpy()
    alpha_diff = abs(alpha_simple - alpha_SRH_2D)

    feature_names = [r"$Fr$", r"$\beta$", r"$C_d$"]

    #analyze using random forest
    #using_random_forest(Fr, beta, Cd, alpha_diff, feature_names)

    # analyze using SHAP
    using_SHAP_for_alpha_dff(Fr, beta, Cd, alpha_diff, feature_names)

    # analyze using PDP
    #using_PDP(Fr, beta, Cd, alpha_diff, feature_names)



def compare_SRH_2D_srhFoam():
    """

    :return:
    """

    parameters = np.loadtxt("all_cases_parameters.dat", delimiter=',')

    nTotal = parameters.shape[0]

    if nTotal!=(Fr_n*beta_n*C_A_n):
        print("nTotal is not consistent with nFr*nbeta*nCd.")
        exit()


    # load the SRH-2D Windows data
    with np.load('Fr_beta_C_A_h2prime_h2_alpha_arrays_SRH_2D_Windows.npz') as data:
        iCases_SRH_2D = data['iCases']
        Frs_SRH_2D = data['Frs']
        betas_SRH_2D = data['betas']
        Cds_SRH_2D = data['Cds']
        alpha_results_SRH_2D = data['alpha']
        bConverged_results_SRH_2D = data['bSuccess']

    # load the srhFoam data
    with np.load('Fr_beta_C_A_h2prime_h2_alpha_arrays_srhFoam_no_porosity.npz') as data:
        iCases_srhFoam = data['iCases']
        Frs_srhFoam = data['Frs']
        betas_srhFoam = data['betas']
        Cds_srhFoam = data['Cds']
        alpha_results_srhFoam = data['alpha']
        bConverged_results_srhFoam = data['bSuccess']

    nTotal = Fr_n * beta_n * C_A_n

    if nTotal!=(Frs_SRH_2D.shape[0]*Frs_SRH_2D.shape[1]*Frs_SRH_2D.shape[2]):
        print("nTotal and the dimensions of Frs_SRH_2D results are not consistent.")
        exit()

    if nTotal!=(Frs_srhFoam.shape[0]*Frs_srhFoam.shape[1]*Frs_srhFoam.shape[2]):
        print("nTotal and the dimensions of Frs_srhFoam results are not consistent.")
        exit()

    # make the data to be in 1D arrays
    Frs_1D_SRH_2D = np.zeros(nTotal)
    betas_1D_SRH_2D = np.zeros(nTotal)
    Cds_1D_SRH_2D = np.zeros(nTotal)
    alphas_1D_SRH_2D = np.zeros(nTotal)
    bConverged_1D_SRH_2D = np.zeros(nTotal)
    iCases_1D_SRH_2D = np.zeros(nTotal)

    Frs_1D_srhFoam = np.zeros(nTotal)
    betas_1D_srhFoam = np.zeros(nTotal)
    Cds_1D_srhFoam = np.zeros(nTotal)
    alphas_1D_srhFoam = np.zeros(nTotal)
    bConverged_1D_srhFoam = np.zeros(nTotal)
    iCases_1D_srhFoam = np.zeros(nTotal)

    #combined results: iCase, Fr, beta, Cd, alpha_Win, alpha_Linux, bConverged_Win, bConverged_Linux
    combined_results_SRH_2D_srhFoam=np.zeros((nTotal,8))

    iCase = 0
    for iFr in range(Fr_n):
        for ibeta in range(beta_n):
            for iCd in range(C_A_n):
                Frs_1D_SRH_2D[iCase] = Frs_SRH_2D[iFr,ibeta,iCd]
                betas_1D_SRH_2D[iCase] = betas_SRH_2D[iFr,ibeta,iCd]
                Cds_1D_SRH_2D[iCase] = Cds_SRH_2D[iFr,ibeta,iCd]
                alphas_1D_SRH_2D[iCase] = alpha_results_SRH_2D[iFr, ibeta, iCd]
                bConverged_1D_SRH_2D[iCase] = bConverged_results_SRH_2D[iFr, ibeta, iCd]
                iCases_1D_SRH_2D[iCase] = iCase

                Frs_1D_srhFoam[iCase] = Frs_srhFoam[iFr, ibeta, iCd]
                betas_1D_srhFoam[iCase] = betas_srhFoam[iFr, ibeta, iCd]
                Cds_1D_srhFoam[iCase] = Cds_srhFoam[iFr, ibeta, iCd]
                alphas_1D_srhFoam[iCase] = alpha_results_srhFoam[iFr, ibeta, iCd]
                bConverged_1D_srhFoam[iCase] = bConverged_results_srhFoam[iFr, ibeta, iCd]
                iCases_1D_srhFoam[iCase] = iCase

                combined_results_SRH_2D_srhFoam[iCase, 0] = iCase
                combined_results_SRH_2D_srhFoam[iCase, 1] = Frs_1D_SRH_2D[iCase]
                combined_results_SRH_2D_srhFoam[iCase, 2] = betas_1D_SRH_2D[iCase]
                combined_results_SRH_2D_srhFoam[iCase, 3] = Cds_1D_SRH_2D[iCase]
                combined_results_SRH_2D_srhFoam[iCase, 4] = alphas_1D_SRH_2D[iCase]
                combined_results_SRH_2D_srhFoam[iCase, 5] = alphas_1D_srhFoam[iCase]
                combined_results_SRH_2D_srhFoam[iCase, 6] = bConverged_1D_SRH_2D[iCase]
                combined_results_SRH_2D_srhFoam[iCase, 7] = bConverged_1D_srhFoam[iCase]

                iCase += 1

    np.savetxt("combined_results_SRH_2D_srhFoam.csv", combined_results_SRH_2D_srhFoam, delimiter=",", fmt="%.2f",
               header="iCase,Fr,beta,Cd,alpha_Win,alpha_Linux,bConverged_Win,bConverged_Linux", comments="")

    #exit()

    #make plot
    fig, ax = plt.subplots(figsize=(6, 6))

    # Define a colormap (e.g., 'coolwarm' colormap)
    cmap = plt.get_cmap('coolwarm')

    #define criteria
    SRH_2D_converged_criterion = bConverged_1D_SRH_2D > 0.5
    SRH_2D_diverged_criterion = bConverged_1D_SRH_2D < 0.5
    srhFoam_converged_criterion = bConverged_1D_srhFoam > 0.5
    srhFoam_diverged_criterion = bConverged_1D_srhFoam < 0.5
    converged_criterion_both = SRH_2D_converged_criterion & srhFoam_converged_criterion
    diverged_criterion_either = SRH_2D_diverged_criterion | srhFoam_diverged_criterion

    print(f"SRH_2D_converged_criterion: {np.sum(SRH_2D_converged_criterion)} out of {len(SRH_2D_converged_criterion)}")
    print(f"srhFoam_converged_criterion: {np.sum(srhFoam_converged_criterion)} out of {len(srhFoam_converged_criterion)}")
    print(f"converged_criterion_both: {np.sum(converged_criterion_both)} out of {len(converged_criterion_both)}")
    print(f"diverged_criterion_either: {np.sum(diverged_criterion_either)} out of {len(diverged_criterion_either)}")

    #define convergence/divergence criterion
    converged_criterion=SRH_2D_converged_criterion
    diverged_criterion = SRH_2D_diverged_criterion
    #converged_criterion = srhFoam_converged_criterion
    #diverged_criterion = srhFoam_diverged_criterion
    #converged_criterion = converged_criterion_both
    #diverged_criterion = diverged_criterion_either

    alphas_1D_converged_SRH_2D = alphas_1D_SRH_2D[converged_criterion]
    alphas_1D_diverged_SRH_2D = alphas_1D_SRH_2D[diverged_criterion]
    alphas_1D_converged_srhFoam = alphas_1D_srhFoam[converged_criterion]
    alphas_1D_diverged_srhFoam = alphas_1D_srhFoam[diverged_criterion]

    iCases_1D_converged_SRH_2D = iCases_1D_SRH_2D[converged_criterion]
    iCases_1D_converged_srhFoam = iCases_1D_srhFoam[converged_criterion]

    Frs_1D_converged = Frs_1D_SRH_2D[converged_criterion]
    Frs_1D_diverged = Frs_1D_SRH_2D[diverged_criterion]
    betas_1D_converged = betas_1D_SRH_2D[converged_criterion]
    betas_1D_diverged = betas_1D_SRH_2D[diverged_criterion]
    Cds_1D_converged = Cds_1D_SRH_2D[converged_criterion]
    Cds_1D_diverged = Cds_1D_SRH_2D[diverged_criterion]

    iCases_1D_converged = iCases_1D_SRH_2D[converged_criterion]
    iCases_1D_diverged = iCases_1D_SRH_2D[diverged_criterion]

    # Set the print options to display the full array
    np.set_printoptions(threshold=np.inf)

    converged_parameters = np.column_stack((iCases_1D_converged, Frs_1D_converged,betas_1D_converged,Cds_1D_converged))
    diverged_parameters  = np.column_stack((iCases_1D_diverged, Frs_1D_diverged, betas_1D_diverged, Cds_1D_diverged))

    #print("converged_parameters: iCase, Fr, beta, Cd\n", converged_parameters)
    print("diverged_parameters: iCase, Fr, beta, Cd\n", diverged_parameters)

    # Plot the diagonal line
    ax.plot([0, 1], [0, 1], color='black', linestyle='--')

    scatter_plot = ax.scatter(alphas_1D_converged_SRH_2D, alphas_1D_converged_srhFoam, c=Cds_1D_converged, edgecolors='none', cmap=cmap, marker='o', s=60)  #s=60
    #scatter_plot = ax.scatter(alphas_1D, alphas_1D_srhFoam, c=betas_1D, edgecolors='none', cmap=cmap, marker='o', s=60, alpha=1-bConverged_1D_srhFoam)
    #scatter_plot = ax.scatter(alphas_1D, alphas_1D_srhFoam, c=Frs_1D, edgecolors='none', cmap=cmap, marker='o',
    #                          s=60)  # alpha=Frs_1D
    #scatter_plot = ax.scatter(alphas_1D, alphas_1D_srhFoam, c=betas_1D, edgecolors='none', cmap=cmap, marker='o',
    #                          s=Cds_1D)  # alpha=Frs_1D

    cbar = plt.colorbar(scatter_plot, fraction=0.03,pad=0.04,aspect=40)
    cbar.set_label(r'$\beta$', fontsize=14)
    #cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])  # Set the desired tick locations

    cbar.ax.tick_params(labelsize=12)  # Set tick font size

    # Additional customizations if needed
    cbar.outline.set_linewidth(0.5)  # Adjust colorbar outline width


    # set the limit for the x and y axes
    #ax.set_xlim([0, 1])
    #ax.set_ylim([0, 1])

    ax.set_aspect('equal')

    # set x and y axes label and font size
    ax.set_xlabel(r'$\alpha$ from SRH-2D', fontsize=16)
    ax.set_ylabel(r'$\alpha$ from srhFoam', fontsize=16)

    # show the ticks on both axes and set the font size
    ax.tick_params(axis='both', which='major', labelsize=12)
    # xtick_spacing=1
    # ax.xaxis.set_major_locator(tick.MultipleLocator(xtick_spacing))
    #ax.xaxis.set_major_formatter(tick.FormatStrFormatter('%.1f'))

    # show legend, set its location, font size, and turn off the frame
    # plt.legend(loc='lower right', fontsize=14, frameon=False)
    plt.show()

    fig.savefig("alpha_comparison_SRH_2D_srhFoam.png", dpi=300, bbox_inches='tight', pad_inches=0)

def compare_srhFoam_with_without_porosity():
    """

    :return:
    """

    parameters = np.loadtxt("all_cases_parameters.dat", delimiter=',')

    nTotal = parameters.shape[0]

    if nTotal!=(Fr_n*beta_n*C_A_n):
        print("nTotal is not consistent with nFr*nbeta*nCd.")
        exit()

    # load the srhFoam with porosity data
    with np.load('Fr_beta_C_A_h2prime_h2_alpha_arrays_srhFoam_with_porosity.npz') as data:
        iCases_srhFoam_w_porosity = data['iCases']
        Frs_srhFoam_w_porosity = data['Frs']
        betas_srhFoam_w_porosity = data['betas']
        Cds_srhFoam_w_porosity = data['Cds']
        alpha_results_srhFoam_w_porosity = data['alpha']
        bConverged_results_srhFoam_w_porosity = data['bSuccess']

    # load the srhFoam without porosity data
    with np.load('Fr_beta_C_A_h2prime_h2_alpha_arrays_srhFoam_no_porosity.npz') as data:
        iCases_srhFoam_wo_porosity = data['iCases']
        Frs_srhFoam_wo_porosity = data['Frs']
        betas_srhFoam_wo_porosity = data['betas']
        Cds_srhFoam_wo_porosity = data['Cds']
        alpha_results_srhFoam_wo_porosity = data['alpha']
        bConverged_results_srhFoam_wo_porosity = data['bSuccess']

    nTotal = Fr_n * beta_n * C_A_n

    if nTotal!=(Frs_srhFoam_w_porosity.shape[0]*Frs_srhFoam_w_porosity.shape[1]*Frs_srhFoam_w_porosity.shape[2]):
        print("nTotal and the dimensions of Frs_srhFoam_w_porosity results are not consistent.")
        exit()

    if nTotal!=(Frs_srhFoam_wo_porosity.shape[0]*Frs_srhFoam_wo_porosity.shape[1]*Frs_srhFoam_wo_porosity.shape[2]):
        print("nTotal and the dimensions of Frs_srhFoam_wo_porosity results are not consistent.")
        exit()

    # make the data to be in 1D arrays
    Frs_1D_srhFoam_w_porosity = np.zeros(nTotal)
    betas_1D_srhFoam_w_porosity = np.zeros(nTotal)
    Cds_1D_srhFoam_w_porosity = np.zeros(nTotal)
    alphas_1D_srhFoam_w_porosity = np.zeros(nTotal)
    bConverged_1D_srhFoam_w_porosity = np.zeros(nTotal)
    iCases_1D_srhFoam_w_porosity = np.zeros(nTotal)

    Frs_1D_srhFoam_wo_porosity = np.zeros(nTotal)
    betas_1D_srhFoam_wo_porosity = np.zeros(nTotal)
    Cds_1D_srhFoam_wo_porosity = np.zeros(nTotal)
    alphas_1D_srhFoam_wo_porosity = np.zeros(nTotal)
    bConverged_1D_srhFoam_wo_porosity = np.zeros(nTotal)
    iCases_1D_srhFoam_wo_porosity = np.zeros(nTotal)

    #combined results: iCase, Fr, beta, Cd, alpha_w_porosity, alpha_wo_porosity, bConverged_w_porosity, bConverged_wo_porosity, alpha_diff
    combined_results_srhFoam_w_wo_porosity=np.zeros((nTotal,9))

    iCase = 0
    for iFr in range(Fr_n):
        for ibeta in range(beta_n):
            for iCd in range(C_A_n):
                Frs_1D_srhFoam_w_porosity[iCase] = Frs_srhFoam_w_porosity[iFr,ibeta,iCd]
                betas_1D_srhFoam_w_porosity[iCase] = betas_srhFoam_w_porosity[iFr,ibeta,iCd]
                Cds_1D_srhFoam_w_porosity[iCase] = Cds_srhFoam_w_porosity[iFr,ibeta,iCd]
                alphas_1D_srhFoam_w_porosity[iCase] = alpha_results_srhFoam_w_porosity[iFr, ibeta, iCd]
                bConverged_1D_srhFoam_w_porosity[iCase] = bConverged_results_srhFoam_w_porosity[iFr, ibeta, iCd]
                iCases_1D_srhFoam_w_porosity[iCase] = iCase

                Frs_1D_srhFoam_wo_porosity[iCase] = Frs_srhFoam_wo_porosity[iFr, ibeta, iCd]
                betas_1D_srhFoam_wo_porosity[iCase] = betas_srhFoam_wo_porosity[iFr, ibeta, iCd]
                Cds_1D_srhFoam_wo_porosity[iCase] = Cds_srhFoam_wo_porosity[iFr, ibeta, iCd]
                alphas_1D_srhFoam_wo_porosity[iCase] = alpha_results_srhFoam_wo_porosity[iFr, ibeta, iCd]
                bConverged_1D_srhFoam_wo_porosity[iCase] = bConverged_results_srhFoam_wo_porosity[iFr, ibeta, iCd]
                iCases_1D_srhFoam_wo_porosity[iCase] = iCase

                combined_results_srhFoam_w_wo_porosity[iCase, 0] = iCase
                combined_results_srhFoam_w_wo_porosity[iCase, 1] = Frs_1D_srhFoam_w_porosity[iCase]
                combined_results_srhFoam_w_wo_porosity[iCase, 2] = betas_1D_srhFoam_w_porosity[iCase]
                combined_results_srhFoam_w_wo_porosity[iCase, 3] = Cds_1D_srhFoam_w_porosity[iCase]
                combined_results_srhFoam_w_wo_porosity[iCase, 4] = alphas_1D_srhFoam_w_porosity[iCase]
                combined_results_srhFoam_w_wo_porosity[iCase, 5] = alphas_1D_srhFoam_wo_porosity[iCase]
                combined_results_srhFoam_w_wo_porosity[iCase, 6] = bConverged_1D_srhFoam_w_porosity[iCase]
                combined_results_srhFoam_w_wo_porosity[iCase, 7] = bConverged_1D_srhFoam_wo_porosity[iCase]
                combined_results_srhFoam_w_wo_porosity[iCase, 8] = abs(alphas_1D_srhFoam_w_porosity[iCase] - alphas_1D_srhFoam_wo_porosity[iCase])

                iCase += 1

    np.savetxt("combined_results_srhFoam_w_wo_porosity.csv", combined_results_srhFoam_w_wo_porosity, delimiter=",", fmt="%.2f",
               header="iCase,Fr,beta,Cd,alpha_w_porosity,alpha_wo_porosity,bConverged_w_porosity,bConverged_wo_porosity,alpha_diff", comments="")

    #exit()

    #make plot
    fig, ax = plt.subplots(figsize=(6, 6))

    # Define a colormap (e.g., 'coolwarm' colormap)
    cmap = plt.get_cmap('coolwarm')

    #define criteria
    srhFoam_w_porosity_converged_criterion = bConverged_1D_srhFoam_w_porosity > 0.5
    srhFoam_w_porosity_diverged_criterion = bConverged_1D_srhFoam_w_porosity < 0.5
    srhFoam_wo_porosity_converged_criterion = bConverged_1D_srhFoam_wo_porosity > 0.5
    srhFoam_wo_porosity_diverged_criterion = bConverged_1D_srhFoam_wo_porosity < 0.5
    converged_criterion_both = srhFoam_w_porosity_converged_criterion & srhFoam_wo_porosity_converged_criterion
    diverged_criterion_either = srhFoam_w_porosity_diverged_criterion | srhFoam_wo_porosity_diverged_criterion

    print(f"srhFoam_w_porosity_converged_criterion: {np.sum(srhFoam_w_porosity_converged_criterion)} out of {len(srhFoam_w_porosity_converged_criterion)}")
    print(f"srhFoam_wo_porosity_converged_criterion: {np.sum(srhFoam_wo_porosity_converged_criterion)} out of {len(srhFoam_wo_porosity_converged_criterion)}")
    print(f"converged_criterion_both: {np.sum(converged_criterion_both)} out of {len(converged_criterion_both)}")
    print(f"diverged_criterion_either: {np.sum(diverged_criterion_either)} out of {len(diverged_criterion_either)}")

    #define convergence/divergence criterion
    #converged_criterion=srhFoam_w_porosity_converged_criterion
    #diverged_criterion = srhFoam_w_porosity_diverged_criterion
    #converged_criterion = srhFoam_wo_porosity_converged_criterion
    #diverged_criterion = srhFoam_wo_porosity_diverged_criterion
    converged_criterion = converged_criterion_both
    diverged_criterion = diverged_criterion_either

    alphas_1D_converged_srhFoam_w_porosity = alphas_1D_srhFoam_w_porosity[converged_criterion]
    alphas_1D_diverged_srhFoam_w_porosity = alphas_1D_srhFoam_w_porosity[diverged_criterion]
    alphas_1D_converged_srhFoam_wo_porosity = alphas_1D_srhFoam_wo_porosity[converged_criterion]
    alphas_1D_diverged_srhFoam_wo_porosity = alphas_1D_srhFoam_wo_porosity[diverged_criterion]

    iCases_1D_converged_srhFoam_w_porosity = iCases_1D_srhFoam_w_porosity[converged_criterion]
    iCases_1D_converged_srhFoam_wo_porosity = iCases_1D_srhFoam_wo_porosity[converged_criterion]

    Frs_1D_converged = Frs_1D_srhFoam_w_porosity[converged_criterion]
    Frs_1D_diverged = Frs_1D_srhFoam_w_porosity[diverged_criterion]
    betas_1D_converged = betas_1D_srhFoam_w_porosity[converged_criterion]
    betas_1D_diverged = betas_1D_srhFoam_w_porosity[diverged_criterion]
    Cds_1D_converged = Cds_1D_srhFoam_w_porosity[converged_criterion]
    Cds_1D_diverged = Cds_1D_srhFoam_w_porosity[diverged_criterion]

    iCases_1D_converged = iCases_1D_srhFoam_w_porosity[converged_criterion]
    iCases_1D_diverged = iCases_1D_srhFoam_w_porosity[diverged_criterion]

    # Set the print options to display the full array
    np.set_printoptions(threshold=np.inf)

    converged_parameters = np.column_stack((iCases_1D_converged, Frs_1D_converged,betas_1D_converged,Cds_1D_converged))
    diverged_parameters  = np.column_stack((iCases_1D_diverged, Frs_1D_diverged, betas_1D_diverged, Cds_1D_diverged))

    #print("converged_parameters: iCase, Fr, beta, Cd\n", converged_parameters)
    print("diverged_parameters: iCase, Fr, beta, Cd\n", diverged_parameters)

    # Plot the diagonal line
    ax.plot([0, 1], [0, 1], color='black', linestyle='--')

    scatter_plot = ax.scatter(alphas_1D_converged_srhFoam_w_porosity, alphas_1D_converged_srhFoam_wo_porosity, c=Cds_1D_converged, edgecolors='none', cmap=cmap, marker='o', s=60)  #s=60
    #scatter_plot = ax.scatter(alphas_1D, alphas_1D_srhFoam, c=betas_1D, edgecolors='none', cmap=cmap, marker='o', s=60, alpha=1-bConverged_1D_srhFoam)
    #scatter_plot = ax.scatter(alphas_1D, alphas_1D_srhFoam, c=Frs_1D, edgecolors='none', cmap=cmap, marker='o',
    #                          s=60)  # alpha=Frs_1D
    #scatter_plot = ax.scatter(alphas_1D, alphas_1D_srhFoam, c=betas_1D, edgecolors='none', cmap=cmap, marker='o',
    #                          s=Cds_1D)  # alpha=Frs_1D

    cbar = plt.colorbar(scatter_plot, fraction=0.03,pad=0.04,aspect=40)
    cbar.set_label(r'$C_d$', fontsize=14)
    #cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])  # Set the desired tick locations

    cbar.ax.tick_params(labelsize=12)  # Set tick font size

    # Additional customizations if needed
    cbar.outline.set_linewidth(0.5)  # Adjust colorbar outline width


    # set the limit for the x and y axes
    #ax.set_xlim([0, 1])
    #ax.set_ylim([0, 1])

    ax.set_aspect('equal')

    # set x and y axes label and font size
    ax.set_xlabel(r'$\alpha$ from srhFoam (with porosity)', fontsize=16)
    ax.set_ylabel(r'$\alpha$ from srhFoam (without porosity)', fontsize=16)

    # show the ticks on both axes and set the font size
    ax.tick_params(axis='both', which='major', labelsize=12)
    # xtick_spacing=1
    # ax.xaxis.set_major_locator(tick.MultipleLocator(xtick_spacing))
    #ax.xaxis.set_major_formatter(tick.FormatStrFormatter('%.1f'))

    # show legend, set its location, font size, and turn off the frame
    # plt.legend(loc='lower right', fontsize=14, frameon=False)
    plt.show()

    fig.savefig("alpha_comparison_srhFoam_w_wo_porosity.png", dpi=300, bbox_inches='tight', pad_inches=0)

if __name__ == "__main__":

    #solve the conceptual model for LWD in open channel flows
    #solve_LWD()

    #Test: example solve: solve_LWD_for_given_Fr_beta_C_A(Fr, beta, C_A, bLWDMomentumChange)
    #solve_LWD_for_given_Fr_beta_C_A(0.2, 0.5, 10, True)
    #solve_LWD_for_given_Fr_beta_C_A(0.45, 0.55, 26.7, True)

    #fit a NN model for alpha(Fr, beta, C_A) and perform analysis
    #fit_flow_split_NN_model()

    #sensitivity analysis
    #sensitivity_analysis()

    # compare SRH-2D and srhFoam
    #compare_SRH_2D_srhFoam()

    # compare simple model with srhFoam
    #compare_simple_model_with_srhFoam()

    # compare simple model with SRH-2D
    #compare_simple_model_with_SRH_2D()

    #nonlinear_regression()
    #symbolic_regression()
    polynomial_regression()

    # analyze the feature dependence of difference between simple model and SRH-2D results
    # diff_alpha = f(Fr,beta,Cd)
    #feature_dependence_alpha_diff_simple_model_with_SRH_2D()

    # analyze the feature dependencde of alpha on Fr, beta, and Cd
    #feature_dependence_alpha_SRH_2D()

    # compare srhFoam: with and without porosity in the SWE equations
    # compare_simple_model_with_SRH_2D()

    #compare_srhFoam_with_without_porosity()

    print("All done!")
