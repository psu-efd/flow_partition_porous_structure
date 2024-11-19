import numpy as np
from matplotlib import pyplot as plt
import json

import os
import sys

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
import threading
from multiprocessing import current_process

import argparse

import LWD_module

plt.rc('text', usetex=True)  #allow the use of Latex for math expressions and equations
plt.rc('font', family='serif') #specify the default font family to be "serif"

#define some global variables
Fr_range = [0.05, 0.95]
beta_range = [0.05, 0.95]
Cd_LWD_range = [0.0, 80.0]
nFr = 10
nbeta = 10
nCd = 10

def generate_case_parameters():
    """
    Generate parameters for cases
    :return:
    """

    #total number of cases
    nTotal = nFr*nbeta*nCd

    Frs = np.linspace(Fr_range[0], Fr_range[1], nFr)
    betas = np.linspace(beta_range[0], beta_range[1], nbeta)
    Cds = np.linspace(Cd_LWD_range[0], Cd_LWD_range[1], nCd)

    parameters = np.zeros([nTotal, 3])

    iCase = 0
    for Fr in Frs:
        for beta in betas:
            for Cd in Cds:
                parameters[iCase, 0] = Fr
                parameters[iCase, 1] = beta
                parameters[iCase, 2] = Cd
                iCase += 1

    np.savetxt("all_cases_parameters.dat", parameters, fmt='%.2f', delimiter=',')

def run_case_with_given_parameters(iCase, parameters):
    """
    Run a case with the given parameter values
    :param iCase:
    :param parameters:
    :return:
    """

    #data = np.load('diverged_case_IDs.npz')

    #diverse_case_IDs = data['diverged_case_IDs']

    #if iCase not in diverse_case_IDs:
    #    return

    process_name = current_process().name
    print(f"Simulation on worker (process) name: {process_name}")

    print(f"{process_name}: iCase = {iCase}")
    print(f"{process_name}: parameters = {parameters[iCase,:]}")

    Fr = parameters[iCase, 0]
    beta = parameters[iCase, 1]
    Cd_LWD = parameters[iCase, 2]

    # run the case with the given parameter values
    caseName = 'case_'+str(iCase).zfill(4)

    result_dict = LWD_module.run_model_with_given_parameters(iCase, Fr, \
            beta, Cd_LWD, destination_folder=caseName)

    #print("result_dict = ", result_dict)

    # Convert ndarray to list
    for key, value in result_dict.items():
        if isinstance(value, np.ndarray):
            result_dict[key] = value.tolist()

    # Save dictionary to a JSON file
    fileName = "results/case_" + str(iCase).zfill(4)+"_result.json"
    with open(fileName, 'w') as json_file:
        json.dump(result_dict, json_file, indent=4)

def run_cases(nCPUs, bParallel=False):
    """
    Run the cases
    :return:
    """

    print(f"Number of CPUs specified: ", nCPUs)

    parameters = np.loadtxt("all_cases_parameters.dat", delimiter=',')

    nTotal = parameters.shape[0]

    #specify the list of cases
    #case_ID_list = list(range(0, nTotal))
    #case_ID_list = list(range(0, 200))
    case_ID_list = [0,1]
    
    print("case_ID_list=", case_ID_list)

    # Create a partial function with the additional parameters
    partial_function = partial(run_case_with_given_parameters, parameters=parameters)

    if not bParallel:
        #run in serial
        for iCase in case_ID_list:
            print("Running case: ", iCase)

            run_case_with_given_parameters(iCase, parameters)
    else:
        with ProcessPoolExecutor(max_workers=nCPUs) as executor:
        #with ThreadPoolExecutor(max_workers=8) as executor:
            print(f"Number of processes used: {executor._max_workers}")
            results = list(executor.map(partial_function, case_ID_list))

def collect_result_from_all_cases():
    """
    collect and combine the results for all cases
    :return:
    """

    parameters = np.loadtxt("all_cases_parameters.dat", delimiter=',')

    nTotal = parameters.shape[0]

    if nTotal!=(nFr*nbeta*nCd):
        print("nTotal is not consistent with nFr*nbeta*nCd.")
        exit()

    # Initialize an empty 3D data cubes to store results
    Frs = np.empty((nFr, nbeta, nCd))
    betas = np.empty((nFr, nbeta, nCd))
    Cds = np.empty((nFr, nbeta, nCd))

    iCases_results = np.empty((nFr, nbeta, nCd))
    h2prime_results = np.empty((nFr, nbeta, nCd))
    h2_results = np.empty((nFr, nbeta, nCd))
    alpha_results = np.empty((nFr, nbeta, nCd))
    bSuccess_results = np.empty((nFr, nbeta, nCd))

    iCase = 0
    for iFr in range(nFr):
        for iBeta in range(nbeta):
            for iCd in range(nCd):
                Frs[iFr,iBeta,iCd] = parameters[iCase,0]
                betas[iFr, iBeta, iCd] = parameters[iCase, 1]
                Cds[iFr, iBeta, iCd] = parameters[iCase, 2]

                json_file_name = "results/case_" + str(iCase).zfill(4) + "_result.json"

                print("json_file_name = ", json_file_name)

                with open(json_file_name, 'r') as file:
                    # Step 3: Load the JSON data into a Python dictionary
                    data = json.load(file)

                iCases_results[iFr, iBeta, iCd] = data["iCase"]
                h2prime_results[iFr,iBeta,iCd] = data["H_p1"]
                h2_results[iFr, iBeta, iCd] = data["H_p3"]
                alpha_results[iFr, iBeta, iCd] = data["Q_fraction"]
                bSuccess_results[iFr, iBeta, iCd] = data["bSuccess"]

                iCase += 1

    # Save arrays in a .npz file (compressed)
    np.savez_compressed('Fr_beta_C_A_h2prime_h2_alpha_arrays_SRH_2D.npz',
                        iCases=iCases_results,
                        Frs=Frs, betas=betas, Cds=Cds,
                        h2prime=h2prime_results, h2=h2_results, alpha=alpha_results, bSuccess=bSuccess_results)

def collect_diverged_cases():
    """
    collect the case IDs of diverged cases
    :return:
    """

    parameters = np.loadtxt("all_cases_parameters.dat", delimiter=',')

    nTotal = parameters.shape[0]

    if nTotal!=(nFr*nbeta*nCd):
        print("nTotal is not consistent with nFr*nbeta*nCd.")
        exit()

    diverged_case_IDs = []

    iCase = 0
    for iFr in range(nFr):
        for iBeta in range(nbeta):
            for iCd in range(nCd):

                json_file_name = "results/case_" + str(iCase).zfill(4) + "_result.json"

                print("json_file_name = ", json_file_name)

                with open(json_file_name, 'r') as file:
                    # Step 3: Load the JSON data into a Python dictionary
                    data = json.load(file)

                if not data["bSuccess"]:
                    diverged_case_IDs.append(iCase)

                iCase += 1

    print("There are ", len(diverged_case_IDs), " diverged cases.")
    print("diverged_case_IDs=", diverged_case_IDs)

    # Save arrays in a .npz file (compressed)
    np.savez_compressed('diverged_case_IDs.npz', diverged_case_IDs=diverged_case_IDs)

if __name__ == "__main__":

    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Process a number from the command line.")

    #generate the parameters for all the cases
    #generate_case_parameters()

    #run cases in serial
    run_cases(1, bParallel=False)

    #or run cases in parallel
    # Add a command-line argument (number)
    #parser.add_argument("nCPUs", type=int, help="An integer number to be processed as the number of CPUs")
    # Parse the arguments from the command line
    #args = parser.parse_args()

    #run_cases(args.nCPUs, bParallel=True)
    #nCPUs = 4
    #run_cases(nCPUs, bParallel=True)

    #collect diverged cases
    #collect_diverged_cases()

    #collect results from all cases
    #collect_result_from_all_cases()

    print("All done!")
