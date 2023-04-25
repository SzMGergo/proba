import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from util import generate_synthetic_data
from trend import fit_linear_regression, visualize_data_and_fit
from sklearn.metrics import mean_squared_error, mean_absolute_error

def calculate_errors(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return mse, mae

def visualize_data_and_fit(x, y, models):
    plt.scatter(x, y)
    for idx, model in enumerate(models):
        y_pred = model['intercept'] + model['coef'] * x.ravel()
        line_style = ['-r', '--g', '-.b', ':m'][idx % 4]
        plt.plot(x, y_pred, line_style, label=f"Model {idx + 1}")
    plt.xlabel("Feature (x)")
    plt.ylabel("Target (y)")
    plt.title("Synthetic Data with Fitted Models")
    plt.legend()
    plt.show()

def main():
    coefficients = [1, 0.02, -0.002, 0.014]
    x_values = np.linspace(-10, 10, 100)
    x, y = generate_synthetic_data(x_values, coefficients)

    model = fit_linear_regression(x, y)

    x_2d = x.reshape(-1, 1)  # Reshape x to a 2D array
    y_pred = model.predict(x_2d)
    mse, mae = calculate_errors(y, y_pred)
    models = [{'intercept': model.intercept_, 'coef': model.coef_[0]}]

    adjustments = [-0.2, -0.1, 0.1, 0.2]
    for adjustment in adjustments:
        modified_intercept = model.intercept_ + adjustment
        modified_coef = model.coef_[0] + adjustment
        models.append({'intercept': modified_intercept, 'coef': modified_coef})

    print("Original fitted parameters:")
    print("Bias (intercept):", model.intercept_)
    print("Weight (coefficient):", model.coef_[0])
    print("Mean Squared Error (MSE):", mse)
    print("Mean Absolute Error (MAE):", mae)
    print("\nModified fitted parameters and errors:")

    for idx, model_dict in enumerate(models[1:], 1):
        mse_modified, mae_modified = calculate_errors(y, model_dict['intercept'] + model_dict['coef'] * x.ravel())
        print(f"\nModel {idx}:")
        print(f"Bias (intercept): {model_dict['intercept']}, Weight (coefficient): {model_dict['coef']}")
        print("Mean Squared Error (MSE):", mse_modified)
        print("Mean Absolute Error (MAE):", mae_modified)

    visualize_data_and_fit(x, y, models)

if __name__ == "__main__":
    main()
