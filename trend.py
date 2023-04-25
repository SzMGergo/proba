import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from util import generate_synthetic_data

def fit_linear_regression(x, y):
    lr = LinearRegression()
    lr.fit(x.reshape(-1, 1), y)
    return lr

def visualize_data_and_fit(x, y, model):
    plt.scatter(x, y)
    plt.xlabel('Feature (x)')
    plt.ylabel('Target (y)')
    plt.title('Synthetic Data with Polynomial Relationship and Noise')

    x_pred = np.linspace(min(x), max(x), len(x)).reshape(-1, 1)
    y_pred = model.predict(x_pred)
    plt.plot(x_pred, y_pred, color='red', label='Linear Regression Fit')
    plt.legend()
    plt.show()

def main():
    coefficients = [1, 0.02, -0.002, 0.014]
    x_values = np.linspace(-10, 10, 100)
    x, y = generate_synthetic_data(x_values, coefficients)
    lr = fit_linear_regression(x, y)
    visualize_data_and_fit(x, y, lr)

if __name__ == '__main__':
    main()

