import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def generate_polinomial_data(coefficients, fromX, toX,
                             n_samples, noise, random_state=None, filepath=None):
    np.random.seed(random_state)

    X = np.random.uniform(fromX, toX, n_samples)
    y = np.polyval(coefficients[::-1], X) + noise * np.random.randn(n_samples)

    if filepath:
        df = pd.DataFrame({'x': X, 'y': y})
        df.to_csv(filepath, index=False, header=False)

    return X.reshape(-1, 1), y

coeffs = [100, 1, 0.2]
X, y = generate_polinomial_data(coeffs, -5, 7, 500, 1, 42, 'data.csv')
plt.scatter(X, y, label='Data', alpha = 0.5)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def plot_train_test_split(X_train, X_test, y_train, y_test):
    plt.scatter(X_train, y_train, label='Train', alpha=0.5)
    plt.scatter(X_test, y_test, label='Test', alpha=0.5)
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Train-Test Split')
    plt.legend()
    plt.show()

plot_train_test_split(X_train, X_test, y_train,y_test)

pass
