import splitting
from sklearn.model_selection import train_test_split

coeffs = [100, 1, 0.2]
X, y = splitting.generate_polinomial_data(coeffs, -5, 7, 500, 1, 42, 'data.csv')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

name, model, mse_on_test, coefficients_on_train_set = \
    splitting.create_train_an_evaluate_polynomial_model(X_train, X_test, y_train, y_test, degree=2)

splitting.hyperparameter_search(X_train, X_test, y_train, y_test,
                          from_degree=1, to_degree=15)


pass