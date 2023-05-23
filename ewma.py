import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

window_size = 100

def create_data():
    filepath = 'input/IVV.csv'
    window_size = 100
    data = pd.read_csv(filepath, parse_dates=['Date'], index_col='Date')
    data['returns'] = np.log(data['Adj Close']).diff()
    data['squared_returns'] = data['returns'] ** 2
    cols = []
    for i in range(1, window_size + 1):
        col = f'lag_{i}'
        data[col] = data['squared_returns'].shift(i)
        cols.append(col)
    data.dropna(inplace=True)
    X = np.array(data[cols])
    y = np.array(data['squared_returns'])
    return X, y

def optimal_decay_factor(X, y, decay_factors):
    best_decay_factor, best_mse = None, float('inf')
    d_mse = {}

    for decay_factor in decay_factors:
        #calculate weights
        weights = decay_factor ** np.arange(window_size)
        weights /= weights.sum()
        #predict data using weights
        X_modified = X[:, :window_size]
        y_pred = np.dot(X_modified, weights)

        #calculate mse
        mse = mean_squared_error(y, y_pred)
        d_mse[decay_factor] = mse
        if mse < best_mse:
            best_decay_factor, best_mse = decay_factor, mse

    for key, value in d_mse.items():
        print(key, value)

    print(f'Best degree: {best_decay_factor}, Best MSE {best_mse}')

X, y = create_data()
decay_factors = np.arange(0.5, 0.99, 0.01)
print(decay_factors)
optimal_decay_factor(X, y, decay_factors)

def hyperparameter_search_linreg(X, y, degrees):
    #splitting to train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

    best_decay_factor, best_mse, best_model = None, float('inf'), None
    d_mse = {}

    for degree in degrees:
        #create linreg model
        model = LinearRegression()
        #fitting model to training set
        X_train_modified = X_train[:, :degree]
        model.fit(X_train_modified, y_train)
        model.coef_, model.intercept_
        #predict model to testset
        y_pred = model.predict(X_test[:, :degree])

        #calculate mse
        mse = mean_squared_error(y_test, y_pred)
        d_mse[degree] = (mse, model.intercept_, model.coef_)
        if mse < best_mse:
            best_degree, best_mse, best_model = degree, mse, model

    for key, value in d_mse.items():
        print(key, value)

    print(f'Best degree: {best_degree}, Best MSE {best_mse}, Best model: {best_model}')

hyperparameter_search_linreg(X, y, range(2, 10))

'''
def print_coeffs(text, model):
    if 'linear_regression' in model.named_steps.keys():
        linreg = 'linear_regression'
    else:
        linreg = 'linearregression'
    coeffs = np.concatenate(([model.named_steps[linreg].intercept_], model.named_steps[linreg].coef_[1:]))
    coeffs_str = ' '.join(np.format_float_positional(coeff, precision=4) for coeff in coeffs)
    print(text + coeffs_str)

#cross validation

def cross_validate(X, y, n_splits=5, from_degree=1, to_degree=10):
    degrees = range(from_degree, to_degree+1)
    kf = KFold(n_splits=n_splits)
    results = {}
    best_model = None
    best_degree = None
    best_mse = np.inf
    np.set_printoptions(precision=4)
    for degree in degrees:
        name, model = create_polynomial_model(degree)
        mse_sum = 0
        for train_idx, val_idx in kf.split(X):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            model, mse = train_and_evaluate_model(model, X_train, y_train, X_val, y_val)
            print_coeffs("Coefficients: ", model)
            mse_sum += mse
        avg_mse = mse_sum / n_splits
        results[degree] = avg_mse
        print(f"for degree: {degree}, MSE: {avg_mse}")
        # fit for the whole dataset
        # model, mse = train_and_evaluate_model(model, X, y, X_val, y_val)1
        model.fit(X, y)
        print_coeffs("Final Coefficients: ", model)
        if avg_mse < best_mse:
            best_mse = avg_mse
            best_degree = degree
            best_model = model
    print(f"Best model: degree={best_degree}, MSE={best_mse}")
    print_coeffs("Coefficients for best model: ", best_model)
    return best_model
'''