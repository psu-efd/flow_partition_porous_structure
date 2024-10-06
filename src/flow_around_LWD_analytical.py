#Analytical solution for open channel hydraulics with a porous large woody debris (LWD)
#The solution is for the flow split (percentage of flow goes through the opening and LWD).

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.optimize import least_squares


plt.rc('text', usetex=True)  #allow the use of Latex for math expressions and equations
plt.rc('font', family='serif') #specify the default font family to be "serif"

# Define the system of nonlinear equations
def equations(vars,Q,h,W,U,beta,C_A,g,bLWDMomentumChange):
    hu, h2, alpha = vars

    eq1 = h2 + alpha**2*Q**2/2/g/h2**2/beta**2/W**2 - (h + U**2/2/g)                  # equation 1
    eq2 = hu**2 - h2**2 - C_A/g/h2 * ( (1-alpha)*Q/(1-beta)/W )**2                    # equation 2

    if bLWDMomentumChange: #if include the momentum change through LWD
        eq2 = eq2 + 2/g* (hu*((1-alpha)*Q/hu/(1-beta)/W)**2)*(1-hu/h2)

    eq3 = hu + 1.0/2/g* (1-alpha)**2 * Q**2 /hu**2/(1-beta)**2/W**2 - (h + U**2/2/g)     # equation 3

    print("residuals = ", eq1,eq2,eq3)

    return [eq1, eq2, eq3]

def solve_LWD():
    """
    Solve the flow around porous LWD problem.

    Parameters
    ----------
    LWD

    Returns
    -------

    """

    #define parameters
    Q = 0.119  #total incoming flow discharge (m^3/s)
    h = 0.48  #upstream incoming flow depth (m)
    W = 1.5  #channel width (m; assumed constant)
    U = Q/(h*W)   #upstream incoming flow velocity (m/s)
    W_LWD = 0.75  #width of LWD
    beta = 1.0 - W_LWD/W   #fraction of opening width w.r.t. channel width
    L_LWD = 0.2 #length of LWD in the streamwise direction (m)

    C_D = 1.0  #drag coefficient on component branches (cylinders) inside LWD
    porosity_LWD = 0.6  #porosity of LWD
    d_LWD = 0.04 #average diameter of branches in LWD
    a = 4.0*(1-porosity_LWD)/3.14/d_LWD  #average frontal area per LWD volume (1/m; see Nepf, 2012)

    C_A = L_LWD*C_D*a/porosity_LWD**3.0

    print("C_A = ", C_A)

    C_A = 10.0

    g = 9.81 #constant for gravity

    #whether to consider the momentum change within LWD
    bLWDMomentumChange = False

    #unknowns: hu, h2, alpha
    initial_guess = [h, h, 0.5]

    # Define bounds for the variables [hu, h2, alpha]
    # Lower bounds
    #lower_bounds = [h-0.1, h-0.1, 0]
    lower_bounds = [0.01, 0.01, 0]
    # Upper bounds
    #upper_bounds = [h+0.1, h+0.1, 1.0]
    #upper_bounds = [1.0, 1.0, 1.0]
    upper_bounds = [2*h, 2*h, 1.0]   #assume the water depht upper bound is two times the upstream water depth

    #solve with fsolve
    #solution = fsolve(equations, initial_guess, args=(Q,h,W,U,beta,C_A,g,bLWDMomentumChange))
    # Display the solution
    #print(f"Solution: hu = {solution[0]}, h2 = {solution[1]}, alpha = {solution[2]}")

    #solve with least_squares
    result = least_squares(equations, initial_guess, bounds=(lower_bounds, upper_bounds), args=(Q,h,W,U,beta,C_A,g,bLWDMomentumChange))
    # Extract the solution
    solution = result.x

    # Check if the optimization was successful
    if result.success:
        # Display the solution
        print(f"Solution: hu = {solution[0]}, h2 = {solution[1]}, alpha = {solution[2]}")
    else:
        print("Optimization failed:", result.message)

    print("result = ", result)

if __name__ == "__main__":

    solve_LWD()

    print("All done!")
