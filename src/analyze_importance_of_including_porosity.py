#Analytical solution for open channel hydraulics with a porous large woody debris (LWD)
#The solution is for the flow split (percentage of flow goes through the opening and LWD).

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import plot_partial_dependence
import shap

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

def analyze_the_importance_of_porosity():
    """
    This function analyzies the importance of whether the include the porosity in the SWEs.

    :return:
    """

    #read data: the csv file should have been created before calling this function.
    data = pd.read_csv('combined_results_srhFoam_w_wo_porosity.csv')

    print(data.head())

    Fr = data['Fr'].to_numpy()
    beta = data['beta'].to_numpy()
    Cd = data['Cd'].to_numpy()
    alpha_w_porosity = data['alpha_w_porosity'].to_numpy()
    alpha_w0_porosity = data['alpha_wo_porosity'].to_numpy()
    alpha_diff = data['alpha_diff'].to_numpy()

    #analyze using random forest
    #using_random_forest(Fr, beta, Cd, alpha_diff)

    # analyze using SHAP
    #using_SHAP(Fr, beta, Cd, alpha_diff)

    # analyze using PDP
    using_PDP(Fr, beta, Cd, alpha_diff)

def using_random_forest(Fr, beta, Cd, alpha_diff):
    """
    UUsing a Random Forest Regressor for Feature Importance

    :param Fr:
    :param beta:
    :param Cd:
    :param alpha_diff:
    :return:
    """

    X = np.column_stack((Fr, beta, Cd))
    y = alpha_diff

    # Fit Random Forest Regressor
    model = RandomForestRegressor()
    model.fit(X, y)

    # Get feature importances
    importances = model.feature_importances_
    feature_names = ['Fr', 'beta', 'Cd']

    # Plot feature importances
    plt.barh(feature_names, importances)
    plt.xlabel("Feature Importance")
    plt.title("Random Forest Feature Importance for alpha_diff=f(Fr, beta, Cd)")
    plt.show()

def using_SHAP(Fr, beta, Cd, alpha_diff):
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
    y = alpha_diff

    # Fit Random Forest Regressor
    model = RandomForestRegressor()
    model.fit(X, y)

    # Use SHAP to explain predictions
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    feature_names = ['Fr', 'beta', 'Cd']

    # Plot SHAP summary plot
    shap.summary_plot(shap_values, X, feature_names=feature_names)

def using_PDP(Fr, beta, Cd, alpha_diff):
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
    y = alpha_diff

    # Fit Random Forest Regressor
    model = RandomForestRegressor()
    model.fit(X, y)

    feature_names = ['Fr', 'beta', 'Cd']

    # Plot partial dependence plots
    plot_partial_dependence(model, X, features=[0, 1, 2], feature_names=feature_names)
    plt.show()


if __name__ == "__main__":

    analyze_the_importance_of_porosity()

    print("All done!")
