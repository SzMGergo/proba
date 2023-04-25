import matplotlib.pyplot as plt
import numpy as np

def osszead(a, b):
    return a+b

def szorzas(a, b):
    return a*b

def osztas(a,b):
    return a/b

def generate_synthetic_data(x, coefficients, seed=42, noise_std=1):
    np.random.seed(seed)
    y = np.polyval(coefficients[::-1], x) + np.random.normal(0, noise_std, len(x))
    return x, y

def visualize_data(x, y):
    plt.scatter(x, y)
    plt.xlabel('Feature (x)')
    plt.ylabel('Target (y)')
    plt.title('Synthetic Data with Polynomial Relationship and Noise')
    plt.show()

def main():
    coefficients = [1, 0.02, -0.002, 0.014]
    x_values = np.linspace(-10, 10, 100)
    x, y = generate_synthetic_data(x_values, coefficients)
    visualize_data(x, y)

main()