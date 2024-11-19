import numpy as np
from matplotlib import pyplot as plt
import os
import shutil
import pandas as pd
from skopt import gp_minimize, dump, load
from skopt.plots import plot_gaussian_process
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import Matern

import pyHMT2D

#a module to place commonly used functions

#measurement noise level
noise_level = 0.05  #5%

#GP parameters
n_calls=10 #10  # the number of evaluations of f
n_initial_points=5 #5  # the number of random initialization points

# Gloabl lists to store the history of Cd/ManningN values, probe values, and objective function values
Cd_history = []
ManningN_history = []
Objective_function_history = []  #total error
Objective_function_history_Water_Depth = []  #error due to water depth
Objective_function_history_Q_split = []      #error due to flow split
WSE_p1_history = []
WSE_p2_history = []
WSE_p3_history = []
H_p1_history = []
H_p2_history = []
H_p3_history = []
Q_Line2_history = []
Q_Line3_history = []

def modify_ManningN(srhhyro_filename, ManningN):
    """
    Modify the srhhyro file and modify ManningN.

    Parameters
    ----------
    srhhyro_filename
    ManningN

    Returns
    -------

    """

    # Read the srhhydro file
    with open(srhhyro_filename, 'r') as file:
        lines = file.readlines()

    # Find and modify the line that starts with the specified string
    for i, line in enumerate(lines):
        if line.startswith("ManningsN 2"):        #find the line starting with "ManningsN 2"  -> LWD
            new_line_content = "ManningsN 2 " + str(ManningN)   #replace the Manning's n value
            lines[i] = new_line_content + '\n'
            break
    else:
        print(f"Error: No line starts with '{line_start}'.")

    # Write the modified lines back to the same file
    with open(srhhyro_filename, 'w') as file:
        file.writelines(lines)

def modify_Cd(srhhyro_filename, Cd):
    """
    Modify the srhhyro file and modify Cd.

    Parameters
    ----------
    srhhyro_filename
    Cd

    Returns
    -------

    """

    # Read the srhhydro file
    with open(srhhyro_filename, 'r') as file:
        lines = file.readlines()

    # Find and modify the line that starts with the specified string
    for i, line in enumerate(lines):
        if line.startswith("DeckParams 1"):        #find the line starting with "DeckParams 1"  -> LWD
            print("line before replacement: ", line)

            # Split the line into a list
            items = line.split()

            print("items before replace: ", items)

            #replace the Cd value
            items[4] = str(Cd)

            # Create the modfied line by joining the items
            new_line_content = ' '.join(str(item) for item in items)

            print("line after replacement: ", new_line_content)

            lines[i] = new_line_content + '\n'
            break
    else:
        print(f"Error: No line starts with '{line_start}'.")

    # Write the modified lines back to the same file
    with open(srhhyro_filename, 'w') as file:
        file.writelines(lines)

def run_srh_2d_model_with_given_ManningN(ManningN, destination_folder='case_temp'):
    """
    Run SRH-2D with the given Manning's n value for LWD, and extract the simulated results.

    Parameters
    ----------
    ManningN
    destination_folder: the case folder name where the simulated results should be saved.

    Returns
    -------

    """

    print("Run SRH-2D case with given Manning's n: ", ManningN)

    #Copy the "case_base" to "case_temp"
    # Source and destination folder paths
    source_folder = 'case_base'
    #destination_folder = 'case_temp'

    # Check if the destination folder exists
    if os.path.exists(destination_folder):
        #give a warning and then exit
        #print('Destination folder already exists! Please remove it and try again.')
        #exit()
        #or remove the existing folder
        shutil.rmtree(destination_folder)  # Removes the destination folder

    # Copy the entire folder to the new destination
    shutil.copytree(source_folder, destination_folder)

    # Get the current working directory
    original_directory = os.getcwd()
    print(f"Original directory: {original_directory}")

    # Go into the case folder
    os.chdir(destination_folder)
    print(f"Inside directory: {os.getcwd()}")

    # Modify the Manning's n value
    modify_ManningN("LWD_flume_experiments.srhhydro", ManningN)

    #set and run SRH-2D
    version = "3.6.5"
    srh_pre_path = r"C:\Program Files\SMS 13.3 64-bit\python\Lib\site-packages\srh2d_exe\SRH_Pre_Console.exe"
    srh_path = r"C:\Program Files\SMS 13.3 64-bit\python\Lib\site-packages\srh2d_exe\SRH-2D_Console.exe"
    extra_dll_path = r"C:\Program Files\SMS 13.3 64-bit\python\Lib\site-packages\srh2d_exe"

    #create a SRH-2D model instance
    my_srh_2d_model = pyHMT2D.SRH_2D.SRH_2D_Model(version, srh_pre_path,
                       srh_path, extra_dll_path, faceless=False)

    #initialize the SRH-2D model
    my_srh_2d_model.init_model()

    print("Hydraulic model name: ", my_srh_2d_model.getName())
    print("Hydraulic model version: ", my_srh_2d_model.getVersion())

    #open a SRH-2D project
    my_srh_2d_model.open_project("LWD_flume_experiments.srhhydro")

    #run SRH-2D Pre to preprocess the case
    my_srh_2d_model.run_pre_model()

    #run the SRH-2D model's current project
    my_srh_2d_model.run_model(bShowProgress=False)

    my_srh_2d_data = my_srh_2d_model.get_simulation_case()

    # read SRH-2D result in XMDF format (*.h5)
    # Whether the XMDF result is nodal or cell center. In SRH-2D's ".srhhydro" file,
    # the output option for "OutputFormat" can be manually changed before simulation.
    # Options are "XMDF" (results at at nodes), "XMDFC" (results are at cell centers), etc.
    # For example, "OutputFormat XMDFC EN". The following lines show that the SRH-2D simulation
    # was run with "XMDFC" as output format (see the "XMDFC" part of the result file name) and thus
    # we set "bNodal = False".
    bNodal = False

    my_srh_2d_data.readSRHXMDFFile(my_srh_2d_data.get_case_name() + "_XMDFC.h5", bNodal)

    # export the SRH-2D result to VTK: lastTimeStep=True means we only want to deal with the last time step.
    # See the code documentation of outputXMDFDataToVTK(...) for more options. It returns a list of vtk file names.
    vtkFileNameList = my_srh_2d_data.outputXMDFDataToVTK(bNodal, lastTimeStep=True, dir='')

    #close the SRH-2D project
    my_srh_2d_model.close_project()

    #quit SRH-2D
    my_srh_2d_model.exit_model()

    #extract data at monitoring points and lines
    #get SRH-2D case name
    case_name = my_srh_2d_data.get_case_name()
    print("Case name = ", case_name)

    #monitoring point 1
    df = pd.read_csv("Output_MISC/" + case_name + "_PT1.dat", delim_whitespace=True)
    # Access the value in the last row and the fifth column (WSE)
    Zb_p1 = df.iloc[-1, 3]  # bed elevation
    WSE_p1 = df.iloc[-1, 4]  #WSE
    H_p1 = df.iloc[-1, 5]    #water depth


    # monitoring point 2
    df = pd.read_csv("Output_MISC/" + case_name + "_PT2.dat", delim_whitespace=True)
    # Access the value in the last row and the fifth column (WSE)
    Zb_p2 = df.iloc[-1, 3]  # bed elevation
    WSE_p2 = df.iloc[-1, 4]
    H_p2 = df.iloc[-1, 5]

    # monitoring point 3
    df = pd.read_csv("Output_MISC/" + case_name + "_PT3.dat", delim_whitespace=True)
    # Access the value in the last row and the fifth column (WSE)
    Zb_p3 = df.iloc[-1, 3]  # bed elevation
    WSE_p3 = df.iloc[-1, 4]
    H_p3 = df.iloc[-1, 5]

    # no need for monitoring line 1 (it is the center line; Q=0)

    # monitoring line 2
    df = pd.read_csv("Output_MISC/" + case_name + "_LN2.dat", delim_whitespace=True)
    # Access the value in the last row and the second column (Q)
    Q_Line2 = abs(df.iloc[-1, 1])   #get the absolute value

    # monitoring line 3
    df = pd.read_csv("Output_MISC/" + case_name + "_LN3.dat", delim_whitespace=True)
    # Access the value in the last row and the second column (Q)
    Q_Line3 = abs(df.iloc[-1, 1])   #get the absolute value

    # Go back to the original directory
    os.chdir(original_directory)
    print(f"Back to original directory: {os.getcwd()}")

    return WSE_p1, WSE_p2, WSE_p3, H_p1, H_p2, H_p3, Zb_p1, Zb_p2, Zb_p3, Q_Line2, Q_Line3

def run_srh_2d_model_with_given_Cd(Cd, destination_folder='case_temp'):
    """
    Run SRH-2D with the given Cd value for LWD, and extract the simuluated results.

    Parameters
    ----------
    Cd

    Returns
    -------

    """

    #Copy the "case_base" to "case_temp"
    # Source and destination folder paths
    source_folder = 'case_base'

    # Check if the destination folder exists
    if os.path.exists(destination_folder):
        #give a warning and then exit
        #print('Destination folder already exists! Please remove it and try again.')
        #exit()
        #or remove the existing folder
        shutil.rmtree(destination_folder)  # Removes the destination folder

    # Copy the entire folder to the new destination
    shutil.copytree(source_folder, destination_folder)

    # Get the current working directory
    original_directory = os.getcwd()
    print(f"Original directory: {original_directory}")

    # Go into the case folder
    os.chdir(destination_folder)
    print(f"Inside directory: {os.getcwd()}")

    # Modify the Manning's n value
    modify_Cd("LWD_flume_experiments.srhhydro", Cd)

    #set and run SRH-2D
    version = "3.6.5"
    srh_pre_path = r"C:\Program Files\SMS 13.3 64-bit\python\Lib\site-packages\srh2d_exe\SRH_Pre_Console.exe"

    srh_path = r"C:\Program Files\SMS 13.3 64-bit\python\Lib\site-packages\srh2d_exe\SRH-2D_Console.exe"
    #srh_path = r"C:\Users\xzl123\Dropbox\student_thesis\Fikiri\LWD_flow_resistance\SRH_2D_code\SRH_2D_parallel\SRH_2D_parallel-main\code\srh_2d_dev\srh_2d\x64\Release_parallel\srh_2d_parallel.exe"

    extra_dll_path = r"C:\Program Files\SMS 13.3 64-bit\python\Lib\site-packages\srh2d_exe"

    #create a SRH-2D model instance
    my_srh_2d_model = pyHMT2D.SRH_2D.SRH_2D_Model(version, srh_pre_path,
                       srh_path, extra_dll_path, faceless=False)

    #initialize the SRH-2D model
    my_srh_2d_model.init_model()

    print("Hydraulic model name: ", my_srh_2d_model.getName())
    print("Hydraulic model version: ", my_srh_2d_model.getVersion())

    #open a SRH-2D project
    my_srh_2d_model.open_project("LWD_flume_experiments.srhhydro")

    #run SRH-2D Pre to preprocess the case
    my_srh_2d_model.run_pre_model()

    #run the SRH-2D model's current project
    my_srh_2d_model.run_model(bShowProgress=False)

    my_srh_2d_data = my_srh_2d_model.get_simulation_case()

    # read SRH-2D result in XMDF format (*.h5)
    # Whether the XMDF result is nodal or cell center. In SRH-2D's ".srhhydro" file,
    # the output option for "OutputFormat" can be manually changed before simulation.
    # Options are "XMDF" (results at nodes), "XMDFC" (results are at cell centers), etc.
    # For example, "OutputFormat XMDFC EN". The following lines show that the SRH-2D simulation
    # was run with "XMDFC" as output format (see the "XMDFC" part of the result file name) and thus
    # we set "bNodal = False".
    bNodal = False

    my_srh_2d_data.readSRHXMDFFile(my_srh_2d_data.get_case_name() + "_XMDFC.h5", bNodal)

    # export the SRH-2D result to VTK: lastTimeStep=True means we only want to deal with the last time step.
    # See the code documentation of outputXMDFDataToVTK(...) for more options. It returns a list of vtk file names.
    vtkFileNameList = my_srh_2d_data.outputXMDFDataToVTK(bNodal, lastTimeStep=True, dir='')

    #close the SRH-2D project
    my_srh_2d_model.close_project()

    #quit SRH-2D
    my_srh_2d_model.exit_model()

    #extract data at monitoring points and lines
    #get SRH-2D case name
    case_name = my_srh_2d_data.get_case_name()
    print("Case name = ", case_name)

    #monitoring point 1
    df = pd.read_csv("Output_MISC/" + case_name + "_PT1.dat", delim_whitespace=True)
    # Access the value in the last row and the fifth column (WSE)
    Zb_p1 = df.iloc[-1, 3]  # bed elevation
    WSE_p1 = df.iloc[-1, 4]  #WSE
    H_p1 = df.iloc[-1, 5]    #water depth


    # monitoring point 2
    df = pd.read_csv("Output_MISC/" + case_name + "_PT2.dat", delim_whitespace=True)
    # Access the value in the last row and the fifth column (WSE)
    Zb_p2 = df.iloc[-1, 3]  # bed elevation
    WSE_p2 = df.iloc[-1, 4]
    H_p2 = df.iloc[-1, 5]

    # monitoring point 3
    df = pd.read_csv("Output_MISC/" + case_name + "_PT3.dat", delim_whitespace=True)
    # Access the value in the last row and the fifth column (WSE)
    Zb_p3 = df.iloc[-1, 3]  # bed elevation
    WSE_p3 = df.iloc[-1, 4]
    H_p3 = df.iloc[-1, 5]

    # no need for monitoring line 1 (it is the center line; Q=0)

    # monitoring line 2
    df = pd.read_csv("Output_MISC/" + case_name + "_LN2.dat", delim_whitespace=True)
    # Access the value in the last row and the second column (Q)
    Q_Line2 = abs(df.iloc[-1, 1])   #get the absolute value

    # monitoring line 3
    df = pd.read_csv("Output_MISC/" + case_name + "_LN3.dat", delim_whitespace=True)
    # Access the value in the last row and the second column (Q)
    Q_Line3 = abs(df.iloc[-1, 1])   #get the absolute value

    # Go back to the original directory
    os.chdir(original_directory)
    print(f"Back to original directory: {os.getcwd()}")

    return WSE_p1, WSE_p2, WSE_p3, H_p1, H_p2, H_p3, Zb_p1, Zb_p2, Zb_p3, Q_Line2, Q_Line3


# Define the objective function to be minimized (for ManningN)
def objective_function_ManningN(ManningN, bFullWidth):
    # Append the current parameter value to the history
    ManningN_history.append(ManningN[0])

    # Run the case with the current Cd value
    WSE_p1, WSE_p2, WSE_p3, H_p1, H_p2, H_p3, Zb_p1, Zb_p2, Zb_p3, Q_Line2, Q_Line3 = \
        run_srh_2d_model_with_given_ManningN(ManningN[0])

    WSE_p1_history.append(WSE_p1)
    WSE_p2_history.append(WSE_p2)
    WSE_p3_history.append(WSE_p3)
    H_p1_history.append(H_p1)
    H_p2_history.append(H_p2)
    H_p3_history.append(H_p3)
    Q_Line2_history.append(Q_Line2)
    Q_Line3_history.append(Q_Line3)

    # read the measurement data
    df_experiment = pd.read_csv('measurement_results.dat', delim_whitespace=True)

    # Access the measurement values
    WSE_upstream = df_experiment['WSE_upstream'].values[0]
    WSE_downstream = df_experiment['WSE_downstream'].values[0]

    H_upstream = WSE_upstream - Zb_p3       #monitoring point #3 is the upstream point
    H_downstream = WSE_downstream - Zb_p2   #monitoring point #2 is the downstream point

    Q_split_exp = df_experiment['Flow_Fraction_open'].values[0]

    objective_function_value_Water_Depth, objective_function_value_Q_split = \
        compute_error(bFullWidth, H_upstream, H_downstream, H_p3, H_p2, Q_split_exp, Q_Line2, Q_Line3)

    objective_function_value = objective_function_value_Water_Depth + objective_function_value_Q_split

    Objective_function_history.append(objective_function_value)
    Objective_function_history_Water_Depth.append(objective_function_value_Water_Depth)
    Objective_function_history_Q_split.append(objective_function_value_Q_split)

    return objective_function_value


# Define the objective function to be minimized (for Cd)
def objective_function_Cd(Cd, bFullWidth):
    # Append the current parameter value to the history
    Cd_history.append(Cd[0])

    # Run the case with the current Cd value
    WSE_p1, WSE_p2, WSE_p3, H_p1, H_p2, H_p3, Zb_p1, Zb_p2, Zb_p3, Q_Line2, Q_Line3 = run_srh_2d_model_with_given_Cd(Cd[0])

    WSE_p1_history.append(WSE_p1)
    WSE_p2_history.append(WSE_p2)
    WSE_p3_history.append(WSE_p3)
    H_p1_history.append(H_p1)
    H_p2_history.append(H_p2)
    H_p3_history.append(H_p3)
    Q_Line2_history.append(Q_Line2)
    Q_Line3_history.append(Q_Line3)

    # read the measurement data
    df_experiment = pd.read_csv('measurement_results.dat', delim_whitespace=True)

    # Access the values
    WSE_upstream = df_experiment['WSE_upstream'].values[0]
    WSE_downstream = df_experiment['WSE_downstream'].values[0]

    H_upstream = WSE_upstream - Zb_p3       #monitoring point #3 is the upstream point
    H_downstream = WSE_downstream - Zb_p2   #monitoring point #2 is the downstream point

    Q_split_exp = df_experiment['Flow_Fraction_open'].values[0]

    objective_function_value_Water_Depth, objective_function_value_Q_split = \
        compute_error(bFullWidth, H_upstream, H_downstream, H_p3, H_p2, Q_split_exp, Q_Line2, Q_Line3)

    objective_function_value = objective_function_value_Water_Depth + objective_function_value_Q_split

    Objective_function_history.append(objective_function_value)
    Objective_function_history_Water_Depth.append(objective_function_value_Water_Depth)
    Objective_function_history_Q_split.append(objective_function_value_Q_split)

    return objective_function_value

def compute_error(bFullWidth, H_upstream_exp, H_downstream_exp, H_upstream_sim, H_downstream_sim, Q_split_exp, Q_Line2, Q_Line3):
    """
    A function to compute the error between simulation and measurement.

    Parameters
    ----------
    bFullWidth: Whether the case is full width or not (half width)
    H_upstream_exp: Water depth of the upstream from measurement
    H_downstream_exp: Water depth of the downstream from measurement
    H_upstream_sim: Water depth of the upstream from simulation
    H_downstream_sim: Water depth of the downstream from simulation
    Q_split_exp: Flow discharge split from measurement
    Q_Line2: Flow discharge from Line 2 (opening) from simulation (only relevant for half width case)
    Q_Line3: Flow discharge from Line 3 (LWD) from simulation (only relevant for half width case)

    Returns
    -------

    """


    objective_function_value_Water_Depth = abs(H_upstream_exp - H_upstream_sim) / H_upstream_exp + \
                                           abs(H_downstream_exp - H_downstream_sim) / H_downstream_exp

    objective_function_value_Q_split = 0.0
    if not bFullWidth:
        objective_function_value_Q_split = abs(abs(Q_Line2) / (abs(Q_Line2) + abs(Q_Line3)) - Q_split_exp)

    return objective_function_value_Water_Depth, objective_function_value_Q_split

def save_calibration_results(ManningN_or_Cd, calibrated_value):
    """
    Save the calibration results to files: simulation_results.dat, calibrated_ManningN.dat/calibrated_Cd.dat
    Returns
    -------
    ManningN_or_Cd: string, "ManningN" or "Cd"
    calibrated_value: float, the calibrated value for ManningN or Cd

    """

    if ManningN_or_Cd == "ManningN":

        with open('simulation_results.dat', 'w') as f:
            f.write('ManningN error_total error_H error_Q_split H_p1 H_p2 H_p3 Q_Line2 Q_Line3\n')

            for ManningN, error_total, error_H, error_Q_split, H_p1, H_p2, H_p3, Q_Line2, Q_Line3 in zip(
                    ManningN_history,
                    Objective_function_history,
                    Objective_function_history_Water_Depth,
                    Objective_function_history_Q_split,
                    H_p1_history,
                    H_p2_history,
                    H_p3_history,
                    Q_Line2_history,
                    Q_Line3_history):

                sublist = [ManningN, error_total, error_H, error_Q_split, H_p1, H_p2, H_p3, Q_Line2, Q_Line3]

                # Join each sublist into a string with spaces separating elements
                f.write(' '.join(map(str, sublist)) + '\n')

        # write the calibrated ManningN to result file
        with open('calibrated_ManningN.dat', 'w') as file:
            file.write(str(calibrated_value))

    elif ManningN_or_Cd == "Cd":
        with open('simulation_results.dat', 'w') as f:
            f.write('Cd error_total error_H error_Q_split H_p1 H_p2 H_p3 Q_Line2 Q_Line3\n')

            for Cd, error_total, error_H, error_Q_split, H_p1, H_p2, H_p3, Q_Line2, Q_Line3 in zip(
                    Cd_history,
                    Objective_function_history,
                    Objective_function_history_Water_Depth,
                    Objective_function_history_Q_split,
                    H_p1_history,
                    H_p2_history,
                    H_p3_history,
                    Q_Line2_history,
                    Q_Line3_history):
                sublist = [Cd, error_total, error_H, error_Q_split, H_p1, H_p2, H_p3, Q_Line2, Q_Line3]

                # Join each sublist into a string with spaces separating elements
                f.write(' '.join(map(str, sublist)) + '\n')

        # write the calibrated Cd to result file
        with open('calibrated_Cd.dat', 'w') as file:
            file.write(str(calibrated_value))

    else:
        print("Wrong ManningN_or_Cd value. Should be ManningN or Cd.")
        exit()



def plot_optimization_results(ManningN_or_Cd, parameter_bounds):
    """
    plot the Gaussain optimization result.

    :return:
    """

    #read in 'gp_result.pkl' data
    result = load('gp_result.pkl')

    #print(result)
    #exit()

    # Plot the model, sampling points, and confidence interval
    #plot_gaussian_process(result)
    #plt.show()

    # Get the evaluated points (X) and corresponding function values (Y)
    X = np.array(result.x_iters)
    Y = np.array(result.func_vals)

    # Fit a GaussianProcessRegressor on the data to get the prediction
    gp = GaussianProcessRegressor(kernel=1**2 * Matern(length_scale=1, nu=2.5),
                                  n_restarts_optimizer=2, noise=(noise_level) ** 2,
                                  normalize_y=True, random_state=822569775)
    gp.fit(X, Y)

    # Define points to plot the model's prediction
    x_vals = np.linspace(parameter_bounds[0][0], parameter_bounds[0][1], 100).reshape(-1, 1)
    y_pred, sigma = gp.predict(x_vals, return_std=True)

    # Plot the model (prediction) and confidence interval
    plt.figure(figsize=(6, 4))

    # Plot the Gaussian process predicted mean (model)
    plt.plot(x_vals, y_pred, 'g--', label='Predicted model error mean')

    # Plot the confidence interval (±1.96 standard deviations)
    plt.fill_between(x_vals.ravel(),
                     y_pred - 1.96 * sigma,
                     y_pred + 1.96 * sigma,
                     alpha=0.2, color='green',
                     edgecolor='none',
                     label='95\% confidence interval')

    # Plot the actual sampled points
    if ManningN_or_Cd=="ManningN":
        plt.plot(X, Y, 'ro', label="Sampled Manning's $n$ points")
    elif ManningN_or_Cd=="Cd":
        plt.plot(X, Y, 'ro', label="Sampled $C_d$ points")
    else:
        print("Wrong ManningN_or_Cd: ", ManningN_or_Cd)
        exit()

    # Add labels and title
    #plt.title("Gaussian Process Model and Confidence Interval")
    if ManningN_or_Cd=="ManningN":
        plt.xlabel("Manning's $n$", fontsize=16)
    elif ManningN_or_Cd=="Cd":
        plt.xlabel("$C_d$", fontsize=16)
    else:
        print("Wrong ManningN_or_Cd: ", ManningN_or_Cd)
        exit()

    plt.ylabel("Simulation model error", fontsize=16)

    # show the ticks on both axes and set the font size
    plt.tick_params(axis='both', which='major', labelsize=12)

    # show legend, set its location, font size, and turn off the frame
    plt.legend(loc='upper left', fontsize=14, frameon=False)

    # Get the current directory name
    current_directory_name = os.path.basename(os.getcwd())

    # save the figure to file
    plt.savefig(current_directory_name + "_calibration.png", dpi=300, bbox_inches='tight', pad_inches=0.02)

    # Show the plot
    # plt.show()
