#Plot results of some selected cases for visualization

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy.interpolate import griddata

import pandas as pd

import os
import vtk
from vtk.util import numpy_support

plt.rc('text', usetex=True)  #allow the use of Latex for math expressions and equations
plt.rc('font', family='serif') #specify the default font family to be "serif"

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
    cbar.ax.tick_params(labelsize=40)  # Set tick font size

    rectangle = plt.Rectangle((rect_x, rect_y), rect_width, rect_height, linewidth=2, edgecolor="white", facecolor="none")
    ax.add_patch(rectangle)

    #ax.set_xlim(-50.1+0.02, 25.1-0.02)
    #ax.set_ylim(0+0.02, 10-0.02)
    ax.set_xlim(xi.min(), xi.max())
    ax.set_ylim(yi.min(), yi.max())

    ax.set_xlabel("x (m)", fontsize=48)
    ax.set_ylabel("y (m)", fontsize=48)

    ax.tick_params(axis='x', labelsize=40)
    ax.tick_params(axis='y', labelsize=40)

    ax.set_title(rf"$Fr$ = {Fr:.2f}, $\beta$ = {beta:.2f}, $C_d$ = {Cd:.2f}, $\alpha_{{SRH-2D}}$ = {alpha_SRH_2D:.2f}, $\alpha_{{simple}}$ = {alpha_simple:.2f}", fontsize=52)

    # Show the customized plot
    plt.tight_layout()

    fig.savefig("vel_mag_contour_"+ str(case_ID).zfill(4) +".png", dpi=300, bbox_inches='tight', pad_inches=0)

    plt.show()

if __name__ == "__main__":

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


    print("All done!")
