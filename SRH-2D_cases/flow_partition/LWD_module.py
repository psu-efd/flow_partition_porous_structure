import numpy as np
from matplotlib import pyplot as plt
import os
import shutil
import pandas as pd
import subprocess

import pyHMT2D
from pyHMT2D.Misc import gmsh2d_to_srh

#a module to place commonly used functions

def modify_srh_files(Fr, beta, Cd):
    """
    Modify the case's files (srhhyro, srhgeom, and srhmpoint) based on the given Fr, beta, and Cd.

    Parameters
    ----------
    Fr
    beta
    Cd

    Returns
    -------

    """

    #parameters
    h0 = 1.0                 #background (downstream water depth)
    U0 = Fr * np.sqrt(9.81*h0)  #background flow velocity
    q = h0*U0                #specific discharge
    W_factor = 10            #channel width = W_factor * h0
    W = W_factor*h0          #channel width
    W_LWD = (1-beta)*W       #LWD width
    Inlet_Q = q*W                  #total channel discharge

    #length of LWD in streamwise direction
    L0 = 0.2
    L0_half = L0/2

    L1 = h0*50     #domain length 1
    L2 = L0        #domain length 2
    L3 = h0*25     #domain length 3

    #probing locations
    #downstream of LWD
    x_probe_1  =-L0_half-0.1
    y_probe_1 = W_LWD/2

    #upstream of LWD
    x_probe_2 = L0_half+0.1
    y_probe_2 = W_LWD/2

    #middle of the opening
    x_probe_3 = 0.0
    y_probe_3 = W_LWD+beta*W/2

    #near inlet
    x_probe_4 = L3-0.1
    y_probe_4 = W/2

    print("Modifying SRH-2D srhhydro file ...")

    # Read the srhhydro file
    with open("LWD_case.srhhydro", 'r') as file:
        lines = file.readlines()

    # Find and modify the line that starts with the specified string
    for i, line in enumerate(lines):
        if line.startswith("IQParams"):        #find the line starting with "IQParams" to change the discharge Q
            #print("line before replacement: ", line)

            # Split the line into a list
            items = line.split()

            #print("items before replace: ", items)

            #replace the Inlet-Q value
            items[2] = str(Inlet_Q)

            # Create the modfied line by joining the items
            new_line_content = ' '.join(str(item) for item in items)

            #print("line after replacement: ", new_line_content)

            lines[i] = new_line_content + '\n'

        elif line.startswith("DeckParams"):        #find the line for the LWD obstruction
            # print("line before replacement: ", line)

            # Split the line into a list
            items = line.split()

            # print("items before replace: ", items)

            # replace the Cd value
            items[4] = str(Cd)

            # replace the LWD line definition based on LWD width
            items[8] = W_LWD
            items[11] = -0.1

            # Create the modfied line by joining the items
            new_line_content = ' '.join(str(item) for item in items)

            # print("line after replacement: ", new_line_content)

            lines[i] = new_line_content + '\n'

    # Write the modified lines back to the same file
    with open("LWD_case.srhhydro", 'w') as file:
        file.writelines(lines)

    print("Modifying SRH-2D srhgeom file ...")

    #convert gmsh to srhgeom (also add monitoring lines for measuring flow
    # in the opening and through LWD for flow partioning calculation)
    monitoringLine1 = np.zeros((2, 2))       #flow through LWD
    monitoringLine1[0, 0] = 0.0    # xML_start
    monitoringLine1[0, 1] = 0.0    # yML_start
    monitoringLine1[1, 0] = 0.0    # xML_end
    monitoringLine1[1, 1] = W_LWD  # yML_end

    monitoringLine2 = np.zeros((2, 2))        #flow through opening
    monitoringLine2[0, 0] = 0.0    # xML_start
    monitoringLine2[0, 1] = W_LWD  # yML_start
    monitoringLine2[1, 0] = 0.0    # xML_end
    monitoringLine2[1, 1] = W      # yML_end

    monitoringLines = []
    monitoringLines.append(monitoringLine1)
    monitoringLines.append(monitoringLine2)

    convert_gmsh_to_srhgeom(monitoringLines)

    #modify monitoring points file srhmpoint
    print("Modifying SRH-2D srhmpoint file ...")

    # Read the srhmpoint file
    with open("LWD_case.srhmpoint", 'r') as file:
        lines = file.readlines()

    # Find and modify the line that starts with the specified string
    for i, line in enumerate(lines):
        if line.startswith("monitorpt 1"):
            # Split the line into a list
            items = line.split()

            # replace values
            items[2] = str(x_probe_1)
            items[3] = str(y_probe_1)

            # Create the modfied line by joining the items
            new_line_content = ' '.join(str(item) for item in items)

            lines[i] = new_line_content + '\n'

        elif line.startswith("monitorpt 2"):
            # Split the line into a list
            items = line.split()

            # replace values
            items[2] = str(x_probe_2)
            items[3] = str(y_probe_2)

            # Create the modfied line by joining the items
            new_line_content = ' '.join(str(item) for item in items)

            lines[i] = new_line_content + '\n'

        elif line.startswith("monitorpt 3"):
            # Split the line into a list
            items = line.split()

            # replace values
            items[2] = str(x_probe_3)
            items[3] = str(y_probe_3)

            # Create the modfied line by joining the items
            new_line_content = ' '.join(str(item) for item in items)

            lines[i] = new_line_content + '\n'

        elif line.startswith("monitorpt 4"):
            # Split the line into a list
            items = line.split()

            # replace values
            items[2] = str(x_probe_4)
            items[3] = str(y_probe_4)

            # Create the modfied line by joining the items
            new_line_content = ' '.join(str(item) for item in items)

            lines[i] = new_line_content + '\n'

    # Write the modified lines back to the same file
    with open("LWD_case.srhmpoint", 'w') as file:
        file.writelines(lines)


def convert_gmsh_to_srhgeom(monitoringLines):
    """
    Convert gmsh to srhgeom. The gmsh mesh has no information for monitoring lines. We need to
    add them as nodeString in the srhgeom file.
    :return:
    """
    gmsh2d_to_srh("LWD_in_channel.msh", "LWD_case", units="Meters",
                  bAddMonitoringLines=True, monitoringLines=monitoringLines)

def parse_probe_vector_data(fileName):
    """
    Parse OpenFOAM's probe data when the variable is a vector such as velocity.
    """

    # Initialize lists to store the scalar values and tuples
    scalars = []
    vectors = []

    # Open and read the file
    with open(fileName, 'r') as file:
        for line in file:
            if line[0]=='#':   #it is a comment
                continue

            # Strip any leading/trailing whitespace
            line = line.strip()

            # Remove the "(" and ")"
            line = line.replace("(", "").replace(")", "")

            # Split the line into individual components
            parts = line.split()

            # First element is the scalar, remaining elements are the components of vectors
            scalar = float(parts[0])
            scalar_vectors = []

            # Iterate over the vector parts (every 3 parts represent one vector, e.g., "0 -7.66666e-15 0")
            for i in range(1, len(parts), 3):
                # Extract the vector as a tuple of three floats, remove the parentheses
                vector = tuple(float(x) for x in parts[i:i+3])
                scalar_vectors.append(vector)

            # Append the scalar and vectors to the lists
            scalars.append(scalar)
            vectors.append(scalar_vectors)

    # Convert lists to NumPy arrays
    scalars_array = np.array(scalars)
    vectors_array = np.array(vectors)

    return scalars_array, vectors_array


def run_model_with_given_parameters(iCase, Fr, beta, Cd, destination_folder):
    """
    Run srhFoam with the given Cd value for LWD, and extract the simuluated results.

    Parameters
    ----------


    Returns
    -------

    """

    bSuccess = True

    #Copy the "case_base"
    # Source and destination folder paths
    source_folder = 'case_base'

    # Get the current working directory
    original_directory = os.getcwd()
    print(f"iCase = {iCase}: Original directory: {original_directory}")

    # Check if the destination folder exists
    if os.path.exists(destination_folder):
        #give a warning and then exit
        #print('Destination folder already exists! Please remove it and try again.')
        #exit()
        #or remove the existing folder
        shutil.rmtree(destination_folder)  # Removes the destination folder

    # Copy the entire folder to the new destination
    shutil.copytree(source_folder, destination_folder)

    # Go into the case folder
    os.chdir(destination_folder)
    print(f"iCase = {iCase}: Inside directory: {os.getcwd()}")

    # Modify the parameter values in SRH-2D files
    modify_srh_files(Fr, beta, Cd)

    #exit()

    bSuccess = True

    # set and run SRH-2D
    version = "3.6.5"
    srh_pre_path = r"C:\Program Files\SMS 13.3 64-bit\python\Lib\site-packages\srh2d_exe\SRH_Pre_Console.exe"
    srh_path = r"C:\Program Files\SMS 13.3 64-bit\python\Lib\site-packages\srh2d_exe\SRH-2D_Console.exe"
    extra_dll_path = r"C:\Program Files\SMS 13.3 64-bit\python\Lib\site-packages\srh2d_exe"

    # create a SRH-2D model instance
    my_srh_2d_model = pyHMT2D.SRH_2D.SRH_2D_Model(version, srh_pre_path,
                                                  srh_path, extra_dll_path, faceless=False)

    # initialize the SRH-2D model
    my_srh_2d_model.init_model()

    print("Hydraulic model name: ", my_srh_2d_model.getName())
    print("Hydraulic model version: ", my_srh_2d_model.getVersion())

    # open a SRH-2D project
    my_srh_2d_model.open_project("LWD_case.srhhydro")

    # run SRH-2D Pre to preprocess the case
    bSuccessTemp = my_srh_2d_model.run_pre_model()

    if not bSuccessTemp:
        bSuccess = False

    # run the SRH-2D model's current project
    bSuccessTemp = my_srh_2d_model.run_model(bShowProgress=False)

    if not bSuccessTemp:
        bSuccess = False

    my_srh_2d_data = my_srh_2d_model.get_simulation_case()

    # read SRH-2D result in XMDF format (*.h5)
    # Whether the XMDF result is nodal or cell center. In SRH-2D's ".srhhydro" file,
    # the output option for "OutputFormat" can be manually changed before simulation.
    # Options are "XMDF" (results at nodes), "XMDFC" (results are at cell centers), etc.
    # For example, "OutputFormat XMDFC EN". The following lines show that the SRH-2D simulation
    # was run with "XMDFC" as output format (see the "XMDFC" part of the result file name) and thus
    # we set "bNodal = False".
    bNodal = False

    try:
        my_srh_2d_data.readSRHXMDFFile(my_srh_2d_data.get_case_name() + "_XMDFC.h5", bNodal)

        # export the SRH-2D result to VTK: lastTimeStep=True means we only want to deal with the last time step.
        # See the code documentation of outputXMDFDataToVTK(...) for more options. It returns a list of vtk file names.
        vtkFileNameList = my_srh_2d_data.outputXMDFDataToVTK(bNodal, lastTimeStep=True, dir='')
    except:
        bSuccess = False
        print(f"iCase = {iCase}: can't convert case_XMDFC.h5 to vtk. The simulation didn't produce the result file.")

    # close the SRH-2D project
    my_srh_2d_model.close_project()

    # quit SRH-2D
    my_srh_2d_model.exit_model()

    # copy the VTK file
    # Source path of the file you want to copy
    vtk_file_list = []
    try:
        vtk_file_list = [file for file in os.listdir('.') if file.endswith('vtk')]
    except:
        vtk_file_list = []
        print(f"iCase = {iCase}: can't get the list of VTK files.")

    if len(vtk_file_list)!=0:
        source_vtk_path = 'SRH2D_LWD_case_C_0003.vtk'
        #source_vtk_path ='VTK/'+vtk_file_list[-1]

        # Destination path where you want to copy the file
        destination_vtk_path = '../results/' + 'case_result_' + str(iCase).zfill(4) + '.vtk' 

        # Copy the file
        # Check if the destination folder exists
        if os.path.exists(source_vtk_path):
            shutil.copy(source_vtk_path, destination_vtk_path)
        else:
            bSuccess=False
            print(f"iCase = {iCase}: VTK file does not exist. Something is wrong.")

    else:
        bSuccess=False
        print(f"iCase = {iCase}: VTK file does not exist. Something is wrong.")


    #extract data at monitoring points and lines
    Zb_p1 = Zb_p2 = Zb_p3 = Zb_p4 = 0.0
    H_p1 = H_p2 = H_p3 = H_p4 = 0.0
    WSE_p1=WSE_p2=WSE_p3=WSE_p4=0.0

    U_p1 = [0.0, 0.0, 0.0]
    U_p2 = [0.0, 0.0, 0.0]
    U_p3 = [0.0, 0.0, 0.0]
    U_p4 = [0.0, 0.0, 0.0]

    Q_Line1=Q_Line2=Q_fraction=0.0

    # extract data at monitoring points and lines
    # get SRH-2D case name
    case_name = my_srh_2d_data.get_case_name()
    print("Case name = ", case_name)

    try:
        # monitoring point 1
        df = pd.read_csv("Output_MISC/" + case_name + "_PT1.dat", delim_whitespace=True)
        # Access the value in the last row and the fifth column (WSE)
        Zb_p1 = df.iloc[-1, 3]   # bed elevation
        WSE_p1 = df.iloc[-1, 4]  # WSE
        H_p1 = df.iloc[-1, 5]    # water depth
        U_p1[0] = df.iloc[-1, 6] # U velocity
        U_p1[1] = df.iloc[-1, 7] # V velocity

        # monitoring point 2
        df = pd.read_csv("Output_MISC/" + case_name + "_PT2.dat", delim_whitespace=True)
        # Access the value in the last row and the fifth column (WSE)
        Zb_p2 = df.iloc[-1, 3]  # bed elevation
        WSE_p2 = df.iloc[-1, 4]
        H_p2 = df.iloc[-1, 5]
        U_p2[0] = df.iloc[-1, 6]  # U velocity
        U_p2[1] = df.iloc[-1, 7]  # V velocity

        # monitoring point 3
        df = pd.read_csv("Output_MISC/" + case_name + "_PT3.dat", delim_whitespace=True)
        # Access the value in the last row and the fifth column (WSE)
        Zb_p3 = df.iloc[-1, 3]  # bed elevation
        WSE_p3 = df.iloc[-1, 4]
        H_p3 = df.iloc[-1, 5]
        U_p3[0] = df.iloc[-1, 6]  # U velocity
        U_p3[1] = df.iloc[-1, 7]  # V velocity

        # monitoring point 4
        df = pd.read_csv("Output_MISC/" + case_name + "_PT4.dat", delim_whitespace=True)
        # Access the value in the last row and the fifth column (WSE)
        Zb_p4 = df.iloc[-1, 3]  # bed elevation
        WSE_p4 = df.iloc[-1, 4]
        H_p4 = df.iloc[-1, 5]
        U_p4[0] = df.iloc[-1, 6]  # U velocity
        U_p4[1] = df.iloc[-1, 7]  # V velocity

        # monitoring line 1
        df = pd.read_csv("Output_MISC/" + case_name + "_LN1.dat", delim_whitespace=True)
        # Access the value in the last row and the second column (Q)
        Q_Line1 = abs(df.iloc[-1, 1])  # get the absolute value

        # monitoring line 2
        df = pd.read_csv("Output_MISC/" + case_name + "_LN2.dat", delim_whitespace=True)
        # Access the value in the last row and the second column (Q)
        Q_Line2 = abs(df.iloc[-1, 1])  # get the absolute value

        Q_fraction = Q_Line2 / (Q_Line1 + Q_Line2)

    except:
        bSuccess=False
        print(f"iCase = {iCase}: An error happended in extracting SRH-2D data. The simulation may have diverged.")

    # Go back to the original directory
    os.chdir(original_directory)
    print(f"iCase = {iCase}: Back to original directory: {os.getcwd()}")

    #remove the case folder
    # Check if the destination folder exists
    if os.path.exists(destination_folder):
        print(f'iCase = {iCase}: Removing case folder: ', destination_folder)
        #shutil.rmtree(destination_folder)  # Removes the destination folder


    # Collect all the results into a dictionary
    result_dict = {
        "iCase": iCase,
        "Fr": Fr,
        "beta": beta,
        "Cd": Cd,
        "bSuccess": bSuccess,
        "WSE_p1": WSE_p1,
        "WSE_p2": WSE_p2,
        "WSE_p3": WSE_p3,
        "WSE_p4": WSE_p4,
        "H_p1": H_p1,
        "H_p2": H_p2,
        "H_p3": H_p3,
        "H_p4": H_p4,
        "Zb_p1": Zb_p1,
        "Zb_p2": Zb_p2,
        "Zb_p3": Zb_p3,
        "Zb_p4": Zb_p4,
        "U_p1": U_p1,
        "U_p2": U_p2,
        "U_p3": U_p3,
        "U_p4": U_p4,
        "Q_Line1": Q_Line1,
        "Q_Line2": Q_Line2,
        "Q_fraction": Q_fraction
    }


    return result_dict


