#Analytical solution for open channel hydraulics with a porous large woody debris (LWD)
#The solution is for the flow split (percentage of flow goes through the opening and LWD).

#Note: This version uses an approach which is different from trational open channel hydraulics analysis for a contraction.
#      This version assumes there is always upstream backwater effect. The downstream water depth is fixed.

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.optimize import least_squares

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model

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

# Define the system of nonlinear equations
def equations(vars, Fr, beta, C_A, bLWDMomentumChange):
    alpha, h1, h2_m, h2_p, h2prime_p = vars

    eq1 = h1 + 1.0/2.0 * Fr**2 / h1**2 - h2_m - alpha**2/beta**2*Fr**2/2/h2_m**2  #equation 1

    eq2 = h2prime_p**2 - h2_p**2 - C_A/h2_p * ( (1-alpha)/(1-beta) )**2 * Fr**2   # equation 2

    if bLWDMomentumChange: #if include the momentum change through LWD
        #eq3 = eq3 + 2/h2prime* ((1-alpha)/(1-beta) )**2 * Fr**2 * (1-h2prime/h2)
        print("Not implemented yet")
        exit()

    eq3 = h1 + 1.0/2.0 * Fr**2 / h1**2 - h2prime_p - 1.0/2.0 * (1-alpha)**2/(1-beta)**2 * Fr**2 / h2prime_p**2   # equation 3

    eq4 = h2_m + 1/2*alpha**2/beta**2*Fr**2/h2_m**2 - 1.0 - Fr**2/2     # equation 4

    eq5 = h2_p + 1/2*(1-alpha)**2/(1-beta)**2*Fr**2/h2_p**2 - 1.0 - Fr**2/2     # equation 5

    print("residuals = ", eq1,eq2,eq3,eq4,eq5)

    return [eq1, eq2, eq3, eq4, eq5]

def jacobian(vars, Fr, beta, C_A, bLWDMomentumChange):
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
    print("Not implemented yet")
    exit()

    alpha, h1, h2_m, h2_p, h2prime_p = vars

    jacobian = np.zeros([4, 4])

    return jacobian


def solve_LWD_for_given_Fr_beta_C_A(Fr, beta, C_A, bLWDMomentumChange):
    """
    Solve the flow around porous LWD problem for a given set of parameters.

    Parameters
    ----------

    Returns
    -------

    """

    solver_option = 1  #1-fsolve, 2-least_squares (try not to use it because it will give a solution no matther the folow is choked or not)

    bConverged = 1 #1-converged (found a good solution); 0-diverged

    if solver_option==1:      #solve with fsolve
        print("Fr, beta, C_A, bLWDMomentumChange=", Fr, beta, C_A, bLWDMomentumChange)

        # There are five unknows and five equations
        # unknowns: alpha, h1, h2_m, h2_p, h2prime_p
        initial_guess = [0.5, 1.0, 1.0, 1.0, 1.0]

        solution, infodict, ier, mesg = fsolve(equations, initial_guess, args=(Fr,beta,C_A,bLWDMomentumChange), full_output=True)
        #solution, infodict, ier, mesg = fsolve(equations, initial_guess, args=(Fr, beta, C_A, bLWDMomentumChange), fprime=jacobian, full_output=True)

        if ier==1:
            bConverged = 1
        else:
            bConverged = 0

        # Display the solution
        print(f"Solution: alpha = {solution[0]}, h1 = {solution[1]}, h2_m = {solution[2]}, h2_p = {solution[3]}, h2prime_p = {solution[4]}")
        # print("solution = ", solution)
        print("    ier=", ier)
        # print("mesg=",mesg)
        print("infodict=",infodict)
        # print("residuals=",np.isclose(equations(solution,Fr,beta,C_A,bLWDMomentumChange),[0,0,0]))
        print("residuals=", equations(solution, Fr, beta, C_A, bLWDMomentumChange))

        # check positivity of solution
        if solution[1] < 0.0 or solution[2] < 0.0 or solution[3] < 0.0 or solution[4] < 0.0:
            bConverged = 0

        #if the solution is not converged, then the flow is choked (no solution)
        if bConverged==0:
            print("    The solution did not converge.")


    elif solver_option==2:     #solve with least_squares

        # unknowns: alpha, h1, h2_m, h2_p, h2prime_p
        initial_guess = [0.5, 1.0, 1.0, 1.0, 1.0]

        # Define bounds for the variables [alpha, h1, h2_m, h2_p, h2prime_p]
        # Lower bounds
        lower_bounds = [0.0, 0.9, 0.5, 0.5, 0.9]
        # Upper bounds
        upper_bounds = [1.0, 2.0, 2.0, 2.0, 2.0]

        result = least_squares(equations, initial_guess, bounds=(lower_bounds, upper_bounds),
                               args=(Fr,beta,C_A,bLWDMomentumChange),
                               #jac=jacobian,
                               method='dogbox')
        # Extract the solution
        solution = result.x

        print("residuals=",equations(solution, Fr, beta, C_A, bLWDMomentumChange))
        print("residuals=", np.isclose(equations(solution, Fr, beta, C_A, bLWDMomentumChange), [0, 0, 0, 0, 0]))

        residuals = np.isclose(equations(solution, Fr, beta, C_A, bLWDMomentumChange), [0, 0, 0, 0, 0])

        # Check if the optimization was successful
        if result.success:
            bConverged = 1
            # Display the solution
            print("Fr, beta, C_A, bLWDMomentumChange=", Fr, beta, C_A, bLWDMomentumChange)
            print(
                f"Solution: alpha = {solution[0]}, h1 = {solution[1]}, h2_m = {solution[2]}, h2_p = {solution[3]}, h2prime_p = {solution[4]}")
        else:
            bConverged = 0
            print("Optimization failed:", result.message)
            print("Fr, beta, C_A, bLWDMomentumChange=", Fr, beta, C_A, bLWDMomentumChange)
            print("result = ", result)
            #exit()

        #check positivity of solution
        if solution[1] < 0.0 or solution[2] < 0.0 or solution[3] < 0.0 or solution[4] < 0.0:
            bConverged = 0

    return solution, bConverged

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
    bLWDMomentumChange = False

    # Initialize an empty 3D data cubes to store results
    iCases_results = np.empty((len(Frs), len(betas), len(C_As)))
    alpha_results = np.empty((len(Frs), len(betas), len(C_As)))
    h1_results = np.empty((len(Frs), len(betas), len(C_As)))
    h2_m_results = np.empty((len(Frs), len(betas), len(C_As)))
    h2_p_results = np.empty((len(Frs), len(betas), len(C_As)))
    h2prime_p_results = np.empty((len(Frs), len(betas), len(C_As)))

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
                solution, bConverged = solve_LWD_for_given_Fr_beta_C_A(Fr, beta, C_A, bLWDMomentumChange)

                # Store the results in the data cubes
                alpha_results[i, j, k] = solution[0]
                h1_results[i, j, k] = solution[1]
                h2_m_results[i, j, k] = solution[2]
                h2_p_results[i, j, k] = solution[3]
                h2prime_p_results[i, j, k] = solution[4]

                bConverged_results[i, j, k] = bConverged

                iCases_results[i, j, k] = iCase

                iCase += 1

    #save results to files
    save_3D_array_to_vtk(alpha_results, "alpha", "alpha_results.vti")
    save_3D_array_to_vtk(h1_results, "h1", "h1_results.vti")
    save_3D_array_to_vtk(h2_m_results, "h2_m", "h2_m_results.vti")
    save_3D_array_to_vtk(h2_p_results, "h2_p", "h2_p_results.vti")
    save_3D_array_to_vtk(h2prime_p_results, "h2prime_p", "h2prime_p_results.vti")
    save_3D_array_to_vtk(bConverged_results, "bConverged", "bConverged_results.vti")

    # Save arrays in a .npz file (compressed)
    print("Saving Fr_beta_C_A_alpha_h1_h2_m_h2_p_h2prime_p_arrays_simple.npz")
    np.savez_compressed('Fr_beta_C_A_alpha_h1_h2_m_h2_p_h2prime_p_arrays_simple.npz',
                        iCases=iCases_results,
                        Frs=Frs, betas=betas, C_As=C_As,
                        alpha=alpha_results, h1=h1_results, h2_m=h2_m_results,
                        h2_p=h2_p_results, h2prime_p=h2prime_p_results,
                        bConverged=bConverged_results)

def fit_flow_split_NN_model():
    """
    Fit the flow split (alpha) as a function of Fr, beta, and C_A with NN
    Returns
    -------

    """

    #load the data
    with np.load('Fr_beta_C_A_h2prime_h2_alpha_arrays_simple.npz') as data:
        Frs = data['Frs']
        betas = data['betas']
        C_As = data['C_As']
        h2prime_results = data['h2prime']
        h2_results = data['h2']
        alpha_results = data['alpha']

    #max_C_A = np.max(C_As)

    #get the lengths of Frs, betas, and C_As arrays
    n_Frs = len(Frs)
    n_betas = len(betas)
    n_C_As = len(C_As)

    # Step 1: build up the dataset
    X = []
    y = []

    # Loop over each combination of Fr, beta, C_A values
    for i in range(len(Frs)):  # Loop over Fr-values
        for j in range(len(betas)):  # Loop over beta-values
            for k in range(len(C_As)):  # Loop over C_A-values
                # Extract the current values of Fr, beta, C_A
                Fr = Frs[i]/Fr_max          #normalize to be in [0, 1]
                beta = betas[j]/beta_max    #normalize to be in [0, 1]
                C_A = C_As[k]/C_A_max       #normalize to be in [0, 1]

                alpha = alpha_results[i, j, k]  #no need to normalize (already in [0, 1])

                X.append([Fr, beta, C_A])
                y.append(alpha)

    # Step 2: Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)

    # Step 3: Build the neural network model
    model = models.Sequential()
    model.add(layers.Input(shape=(3,)))  # Three input variables
    model.add(layers.Dense(32, activation='relu'))  # First hidden layer
    model.add(layers.Dense(32, activation='relu'))  # Second hidden layer
    model.add(layers.Dense(1))  # Output layer (since it's a regression task)

    # Compile the model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Step 4: Train the model
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val))

    # Step 5: Validate the model
    val_loss, val_mae = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Loss: {val_loss}, Validation MAE: {val_mae}")

    # Step 6: Predict on the validation set
    y_pred = model.predict(X_val)
    print("y_pred = ", y_pred)
    print("y_val = ", y_val)
    print("y_error = ", y_pred.ravel()-y_val)

    # Step 7: Save the trained model in the SavedModel format (this will create a directory)
    model.save('alpha_fit_saved_model')

    # Visualize the learning curves
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    #plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.yscale("log")
    plt.legend()
    plt.show()

def sensitivity_analysis():
    """
    With the trained NN model, compute sensitivity (gradients) of output with respect to input variables

    Returns
    -------

    """

    # load the data
    with np.load('Fr_beta_C_A_h2prime_h2_alpha_arrays_simple.npz') as data:
        Frs = data['Frs']
        betas = data['betas']
        C_As = data['C_As']
        alpha_results = data['alpha']

    # build up the dataset
    X = []
    y_truth = []

    # Loop over each combination of Fr, beta, C_A values
    for i in range(len(Frs)):  # Loop over Fr-values
        for j in range(len(betas)):  # Loop over beta-values
            for k in range(len(C_As)):  # Loop over C_A-values
                # Extract the current values of Fr, beta, C_A
                Fr = Frs[i] / Fr_max  # normalize to be in [0, 1]
                beta = betas[j] / beta_max  # normalize to be in [0, 1]
                C_A = C_As[k] / C_A_max  # normalize to be in [0, 1]

                alpha = alpha_results[i, j, k]  # no need to normalize (already in [0, 1])

                X.append([Fr, beta, C_A])
                y_truth.append(alpha)

    # Load the model from the SavedModel directory
    model = load_model('alpha_fit_saved_model')

    # You can now use the loaded model to make predictions
    y_pred = model.predict(X).ravel()

    y_error = y_truth - y_pred   #y_error is a 1D array

    # Convert the input data set to a Tensor for automatic differentiation
    X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)

    # Use GradientTape to compute the gradient of the output with respect to the input
    with tf.GradientTape(persistent=True) as tape:
        # Watch the input variables
        tape.watch(X_tensor)
        # Get model predictions for the validation set
        y_pred = model(X_tensor)

        # Compute the gradients (d_alpha/d_Fr, d_alpha/d_beta, d_alpha/d_C_A)
        gradients = tape.gradient(y_pred, X_tensor)

    # Display the computed gradients (sensitivities)
    gradients_np = gradients.numpy()  # Convert the tensor to a NumPy array

    # Example: Show the first few gradients for each input variable
    print("X_val = ", X)
    print("Gradients of the output with respect to input variables (first few samples):")
    print("d_alpha/d_Fr (Sensitivity to Fr):", gradients_np[:, 0][:5])  # Sensitivity w.r.t. Fr
    print("d_alpha/d_beta (Sensitivity to beta):", gradients_np[:, 1][:5])  # Sensitivity w.r.t. beta
    print("d_alpha/d_C_A (Sensitivity to C_A):", gradients_np[:, 2][:5])  # Sensitivity w.r.t. C_A

    y_error_3D = np.zeros_like(alpha_results)
    d_alpha_d_Fr_3D = np.zeros_like(alpha_results)
    d_alpha_d_beta_3D = np.zeros_like(alpha_results)
    d_alpha_d_C_A_3D = np.zeros_like(alpha_results)

    for i in range(len(Frs)):  # Loop over Fr-values
        for j in range(len(betas)):  # Loop over beta-values
            for k in range(len(C_As)):  # Loop over C_A-values
                y_error_3D[i, j, k] = y_error[i * beta_n * C_A_n + j * beta_n + k]
                d_alpha_d_Fr_3D[i, j, k] = gradients_np[:, 0][i * beta_n * C_A_n + j * beta_n + k]
                d_alpha_d_beta_3D[i, j, k] = gradients_np[:, 1][i * beta_n * C_A_n + j * beta_n + k]
                d_alpha_d_C_A_3D[i, j, k] = gradients_np[:, 2][i * beta_n * C_A_n + j * beta_n + k]

    # save y_error et al. to vtk
    save_3D_array_to_vtk(y_error_3D, "y_error", "y_error.vti")
    save_3D_array_to_vtk(d_alpha_d_Fr_3D, "d_alpha_d_Fr", "d_alpha_d_Fr.vti")
    save_3D_array_to_vtk(d_alpha_d_beta_3D, "d_alpha_d_beta", "d_alpha_d_beta.vti")
    save_3D_array_to_vtk(d_alpha_d_C_A_3D, "d_alpha_d_C_A", "d_alpha_d_C_A.vti")

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
    with np.load('Fr_beta_C_A_h2prime_h2_alpha_arrays_srhFoam.npz') as data:
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

    fig.savefig("alpha_comparison.png", dpi=300, bbox_inches='tight', pad_inches=0)

def compare_simple_model_with_SRH_2D():
    """
    Compare the simple model solution with SRH-2D results.

    :return:
    """

    # load the simple model data
    with np.load('Fr_beta_C_A_alpha_h1_h2_m_h2_p_h2prime_p_arrays_simple.npz') as data:
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
               header="iCase, Fr, beta, Cd, alpha_simple, alpha_SRH_2D, bConverged_simple, bConverged_SRH_2D",
               comments="")

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
    SRH_2D_model_converged_criterion = bConverged_1D_SRH_2D > 0.5
    SRH_2D_model_diverged_criterion = bConverged_1D_SRH_2D < 0.5
    converged_criterion_both = simple_model_converged_criterion & SRH_2D_model_converged_criterion
    diverged_criterion_both = simple_model_diverged_criterion | SRH_2D_model_diverged_criterion

    print(f"simple_model_converged_criterion: {np.sum(simple_model_converged_criterion)} out of {len(simple_model_converged_criterion)}")
    print(f"SRH_2D_model_converged_criterion: {np.sum(SRH_2D_model_converged_criterion)} out of {len(SRH_2D_model_converged_criterion)}")
    print(f"converged_criterion_both: {np.sum(converged_criterion_both)} out of {len(converged_criterion_both)}")

    #define convergence/divergence criterion
    #converged_criterion=simple_model_converged_criterion
    #diverged_criterion = simple_model_diverged_criterion
    #converged_criterion = SRH_2D_model_converged_criterion
    #diverged_criterion = SRH_2D_model_diverged_criterion
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

    converged_parameters = np.column_stack((iCases_1D_converged, Frs_1D_converged,betas_1D_converged,Cds_1D_converged))
    diverged_parameters  = np.column_stack((iCases_1D_diverged, Frs_1D_diverged, betas_1D_diverged, Cds_1D_diverged))

    #print("converged_parameters: iCase, Fr, beta, Cd\n", converged_parameters)
    print("diverged_parameters: iCase, Fr, beta, Cd\n", diverged_parameters)

    # Plot the diagonal line
    ax.plot([0, 1], [0, 1], color='black', linestyle='--')

    scatter_plot = ax.scatter(alphas_1D_converged, alphas_1D_SRH_2D_converged, c=Cds_1D_converged, edgecolors='none', cmap=cmap, marker='o', s=60)  #s=60
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
    ax.set_ylabel(r'$\alpha$ from SRH-2D', fontsize=16)

    # show the ticks on both axes and set the font size
    ax.tick_params(axis='both', which='major', labelsize=12)
    # xtick_spacing=1
    # ax.xaxis.set_major_locator(tick.MultipleLocator(xtick_spacing))
    #ax.xaxis.set_major_formatter(tick.FormatStrFormatter('%.1f'))

    # show legend, set its location, font size, and turn off the frame
    # plt.legend(loc='lower right', fontsize=14, frameon=False)
    plt.show()

    fig.savefig("alpha_comparison_simple_vs_SRH_2D.png", dpi=300, bbox_inches='tight', pad_inches=0)

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
    #solve_LWD_for_given_Fr_beta_C_A(0.15, 0.55, 26.7, False)

    #fit a NN model for alpha(Fr, beta, C_A) and perform analysis
    #fit_flow_split_NN_model()

    #sensitivity analysis
    #sensitivity_analysis()

    # compare SRH-2D and srhFoam
    #compare_SRH_2D_srhFoam()

    # compare simple model with srhFoam
    #compare_simple_model_with_srhFoam()

    # compare simple model with SRH-2D
    compare_simple_model_with_SRH_2D()

    # compare srhFoam: with and without porosity in the SWE equations
    # compare_simple_model_with_SRH_2D()

    #compare_srhFoam_with_without_porosity()

    print("All done!")
