#Plot results of flume experiment cases for visualization
#plot SRH-2D simulation results + measurement velocity profiles

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy.interpolate import griddata, interp2d

import pandas as pd

import os
import vtk
from vtk.util import numpy_support

plt.rc('text', usetex=True)  #allow the use of Latex for math expressions and equations
plt.rc('font', family='serif') #specify the default font family to be "serif"

def plot_contour_from_vtk(case_ID, Fr, beta, Cd, alpha_exp, alpha_simple, rect_x, rect_y, rect_width, rect_height, vtkFileName,
                          U_all, x_positions, y_positions):

    #load data from vtk file: water depht and velocity

    if not os.path.exists(vtkFileName):
        return None

    # Load the VTK file
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(vtkFileName)  # Replace with your VTK file path
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

    #Flip x coordinates
    points[:, 0] = - points[:, 0]

    # Extract values from cell data (assuming the scalar field is at cell centers)
    Velocity_m_p_s = data.GetCellData().GetArray("Velocity_m_p_s")
    Vel_Mag_m_p_s = data.GetCellData().GetArray("Vel_Mag_m_p_s")
    Water_Depth_m = data.GetCellData().GetArray("Water_Depth_m")

    if Velocity_m_p_s is None:
        raise ValueError("No Velocity_m_p_s data found at cell centers. Please check your VTK file.")

    # Convert data to a numpy array
    Vel_x_np = np.array([-Velocity_m_p_s.GetTuple3(i)[0] for i in range(Velocity_m_p_s.GetNumberOfTuples())])  #flip x velocity
    Vel_y_np = np.array([ Velocity_m_p_s.GetTuple3(i)[1] for i in range(Velocity_m_p_s.GetNumberOfTuples())])  # flip x velocity

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

    #plot velocity vectors from experiments

    for iProfile in range(5):
        #How many points of velocity measurement on the profile
        if iProfile == 2:   #the profile at LWD only has 3 points; others have 5.
            nPoints = 3
        else:
            nPoints = 5

        coords_at_points = np.zeros((nPoints,2))

        vel_x_SRH_2D_at_points = np.zeros(nPoints)
        vel_y_SRH_2D_at_points = np.zeros(nPoints)

        for iPoint in range(nPoints):
            coords_at_points[iPoint,0] = x_positions[iProfile]
            coords_at_points[iPoint,1] = y_positions[iProfile][iPoint]

        interpolated_SRH_2D_velocity_x = griddata((x, y), Vel_x_np, coords_at_points, method='linear')
        interpolated_SRH_2D_velocity_y = griddata((x, y), Vel_y_np, coords_at_points, method='linear')

        # Plot the profile
        for iPoint in range(nPoints):
            #plot velocity from experiment
            plt.arrow(coords_at_points[iPoint, 0], coords_at_points[iPoint, 1], U_all[iProfile][iPoint], 0, linewidth=2, head_width=0.04, head_length=0.02, fc='blue', ec='blue', length_includes_head=True)
            plt.plot(coords_at_points[iPoint, 0] + U_all[iProfile][iPoint] + 0.01, coords_at_points[iPoint, 1], 'ko', markersize=4)  # Smaller dot (size 4) to the right of the arrow tip

            #plot SRH-2D velocity
            plt.arrow(coords_at_points[iPoint, 0], coords_at_points[iPoint, 1],
                      interpolated_SRH_2D_velocity_x[iPoint], interpolated_SRH_2D_velocity_y[iPoint],
                      head_width=0.04, head_length=0.02,fc='yellow', ec='yellow',
                      linestyle="--", linewidth=2, length_includes_head=True)

        #plot a velocity vector scale
        plt.arrow(-4, 0.75, 0.5, 0.0,
                  head_width=0.04, head_length=0.02, fc='black', ec='black',
                  linewidth=2, length_includes_head=True)

        plt.text(-3.75, 0.8, "0.5 m/s", color="black", fontsize=36, horizontalalignment='center',)

        dot_x_positions1 = [x_positions[iProfile] + value + 0.01 for value in U_all[iProfile]]
        plt.plot(dot_x_positions1, y_positions[iProfile], 'k-', linewidth=1)  # Curve connecting the dots for profile
        plt.plot([x_positions[iProfile], x_positions[iProfile]], [y_positions[iProfile][0], y_positions[iProfile][-1]], 'k--',
                 linewidth=1)  # Dotted vertical line for profile

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

    #ax.set_title(rf"$Fr$ = {Fr:.2f}, $\beta$ = {beta:.2f}, $C_d$ = {Cd:.2f}, $\alpha_{{SRH-2D}}$ = {alpha_SRH_2D:.2f}, $\alpha_{{simple}}$ = {alpha_simple:.2f}", fontsize=52)
    ax.set_title(
        rf"$Fr$ = {Fr:.3f}, $\beta$ = {beta:.2f}, $C_d$ = {Cd:.1f}, $\alpha_{{exp}}$ = {alpha_exp:.3f}, $\alpha_{{simple}}$ = {alpha_simple:.3f}",
        fontsize=52)

    # Show the customized plot
    plt.tight_layout()

    fig.savefig("exp_vel_mag_contour_"+ str(case_ID).zfill(4) +".png", dpi=300, bbox_inches='tight', pad_inches=0)

    plt.show()

if __name__ == "__main__":

    case_IDs = [1, 2, 3, 4]
    #case_IDs = [1]

    Frs = [0.076, 0.097, 0.126, 0.163]
    Cds = [64.8, 53.4, 45.5, 40]
    betas = [0.5, 0.5, 0.5, 0.5]
    alphas_exp = [0.868, 0.857, 0.846, 0.836]
    alphas_simple = [0.890030526117476, 0.8802260381141247, 0.8714323457381947, 0.8637256205865179]

    #Velocity data from flume experiments
    # First set of values (upstream 72 inches = - 1.83 m)
    U1_case1 = [0.159, 0.15828, 0.14064, 0.17036, 0.1974]    # Experiment 1
    U1_case2 = [0.2058, 0.197, 0.17086, 0.22468, 0.23558]    # Experiment 2
    U1_case3 = [0.260867, 0.200533, 0.174933, 0.211233, 0.202233]    # Experiment 3
    U1_case4 = [0.346067, 0.2741, 0.2496, 0.2393, 0.216467]  # Experiment 4
    y_positions1 = [0.15, 0.45, 0.75, 1.05, 1.35]  # Adjusted y-positions for 0.3 m spacing

    # Second set of values (upstream 36 inches = - 0.91 m)
    U2_case1 = [0.1392, 0.14738, 0.1504, 0.1738, 0.20766]    # Experiment 1
    U2_case2 = [0.18392, 0.18488, 0.18582, 0.22472, 0.25718]    # Experiment 2
    U2_case3 = [0.212133, 0.1949, 0.193467, 0.217833, 0.246067]    # Experiment 3
    U2_case4 = [0.2819, 0.266133, 0.253, 0.2675, 0.270133]  # Experiment 4
    y_positions2 = y_positions1  # Same y-positions for consistency

    # Third set of values (only three values, with the first two removed, obstruction 00 inches = 0 m)
    U3_case1 = [0.1997, 0.30588, 0.30756]    # Experiment 1
    U3_case2 = [0.2494, 0.38214, 0.39144]    # Experiment 2
    U3_case3 = [0.200967, 0.418833, 0.405733]    # Experiment 3
    U3_case4 = [0.244933, 0.538833, 0.4994]  # Experiment 4
    y_positions3 = y_positions1[2:]  # Keeping the positions for the remaining three arrows

    # Fourth set of values (downstream 36 inches = 0.91 m)
    U4_case1 = [0.03402, 0.03252, 0.01858, 0.36968, 0.38428]     # Experiment 1
    U4_case2 = [0.04048, 0.04312, 0.02374, 0.46876, 0.48606]     # Experiment 2
    U4_case3 = [0.0391, 0.042566667, 0.021133, 0.5168, 0.522]     # Experiment 3
    U4_case4 = [0.062, 0.063933, 0.028133, 0.671933, 0.672167]  # Experiment 4
    y_positions4 = y_positions1  # Same y-positions for consistency

    # Fifth set of values (downstream 72 inches = 1.83 m)
    U5_case1 = [0.01384, 0.01244, 0.07918, 0.33488, 0.38092]     # Experiment 1
    U5_case2 = [0.0285, 0.02416, 0.09756, 0.42988, 0.48924]     # Experiment 2
    U5_case3 = [0.0326, 0.033467, 0.111133, 0.4688, 0.529433]     # Experiment 3
    U5_case4 = [0.0514, 0.04067, 0.128533, 0.6009, 0.6785]  # Experiment 4
    y_positions5 = y_positions1  # Same y-positions for consistency

    #put all the exp. data together
    U1s = [U1_case1, U1_case2, U1_case3, U1_case4]
    U2s = [U2_case1, U2_case2, U2_case3, U2_case4]
    U3s = [U3_case1, U3_case2, U3_case3, U3_case4]
    U4s = [U4_case1, U4_case2, U4_case3, U4_case4]
    U5s = [U5_case1, U5_case2, U5_case3, U5_case4]
    x_positions = [-1.83, -0.91, 0, 0.91, 1.83]
    y_positions = [y_positions1, y_positions2, y_positions3, y_positions4, y_positions5]

    #dimensions of the LWD
    rect_x, rect_y = -0.1, 0  # Bottom-left corner of the rectangle
    rect_width =  0.2 # Width of the rectangle
    rect_height = 0.75 # Height of the rectangle

    for i, case_ID in enumerate(case_IDs):
        print("plotting case_ID = ", case_ID)

        vtkFileName = "Exp_"+str(case_ID)+"_Cd/case_final/SRH2D_Exp_"+str(case_ID)+"_Cd_C_0003"".vtk"
        plot_contour_from_vtk(case_ID, Frs[i], betas[i], Cds[i], alphas_exp[i], alphas_simple[i], rect_x, rect_y, rect_width, rect_height, vtkFileName,
                              [U1s[i], U2s[i], U3s[i], U4s[i], U5s[i]], x_positions, y_positions)

    print("All done!")
