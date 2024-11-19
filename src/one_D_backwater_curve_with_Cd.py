#Analytical solution for open channel hydraulics with a porous large woody debris (LWD)
#The solution is for the flow split (percentage of flow goes through the opening and LWD).

import numpy as np
import matplotlib.pyplot as plt

plt.rc('text', usetex=False)  #allow the use of Latex for math expressions and equations
plt.rc('font', family='serif') #specify the default font family to be "serif"


def C_D(x):
    """
    C_D value depends on x

    Parameters
    ----------
    x

    Returns
    -------

    """
    if x >= -0.1 and x <= 0.1:
        C_D = 0.0  #60.0
    else:
        C_D = 0.0

    return C_D

# Define the ODE: dh/dx = C_D q^2 h /2/L0/(q^2 - g h^3)
def f(x, h, q, L0, n):
    return -( C_D(x)*q**2*h/2/L0 + 9.8*n**2*q**2/h**(1.0/3.0) )/(q**2-9.8*h**3)


# Implement the Runge-Kutta 4th order (RK4) method
def runge_kutta_4(f, x0, h0, x_end, delta_x, q, L0, n):
    # Initialize lists to store the x and y values
    x_vals = [x0]
    h_vals = [h0]

    # Iteratively apply RK4 until x reaches x_end
    x = x0
    h = h0
    while x < x_end:
        # Calculate the RK4 steps
        k1 = delta_x * f(x, h, q, L0, n)
        k2 = delta_x * f(x + 0.5 * delta_x, h + 0.5 * k1, q, L0, n)
        k3 = delta_x * f(x + 0.5 * delta_x, h + 0.5 * k2, q, L0, n)
        k4 = delta_x * f(x + delta_x, h + k3, q, L0, n)

        # Update the next value of y and x
        h_next = h + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        x_next = x + delta_x

        # Append the values to the lists
        x_vals.append(x_next)
        h_vals.append(h_next)

        # Update the current values of x and y
        x = x_next
        h = h_next

    return np.array(x_vals), np.array(h_vals)


def predict_correct(f, x0, h0, x_end, delta_x, q, L0, n):
    # Initialize lists to store the x and y values
    x_vals = [x0]
    h_vals = [h0]

    # Iteratively apply predict-correct scheme until x reaches x_end
    x = x0
    h = h0
    while x < x_end:
        # Predictor step: Euler method to predict y at the next step
        h_predict = h + delta_x * f(x, h, q, L0, n)
        x_next = x + delta_x

        # Corrector step: Use the average slope for correction
        h_next = h + (delta_x / 2) * (f(x, h, q, L0, n) + f(x_next, h_predict, q, L0, n))

        # Append the values to the lists
        x_vals.append(x_next)
        h_vals.append(h_next)

        # Update the current values of x and h
        x = x_next
        h = h_next

    return np.array(x_vals), np.array(h_vals)

if __name__ == "__main__":
    # Initial condition y(0) = 1, solving from x=0 to x=5
    x0 = -10
    y0 = 0.44
    x_end = 5
    delta_x = 0.05  # Step size
    q = 0.0753
    L0 = 0.2
    n = 0.02  #Manning n

    # Solve the ODE using the RK4 method
    #x_vals, h_vals = runge_kutta_4(f, x0, y0, x_end, delta_x, q, L0, n)
    x_vals, h_vals = predict_correct(f, x0, y0, x_end, delta_x, q, L0, n)

    print("h = ", h_vals)

    # Plot the results
    plt.plot(x_vals, h_vals, label="Solution")
    #plt.title("Solution of First-Order Nonlinear ODE using RK4")
    plt.xlabel("x")
    plt.ylabel("h")
    plt.grid(True)
    plt.legend()
    plt.show()

    print("All done!")
