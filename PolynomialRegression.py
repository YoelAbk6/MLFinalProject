from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.utils import check_random_state
from utils import save_confusion_matrix
import numpy as np


class polynominal_regression:
    def __init__(self, X_train, X_test, y_train, y_test, rs) -> None:

        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)

        # Fit the model using training data
        poly_reg_model = LinearRegression()
        poly_reg_model.fit(X_train_poly, y_train)

        # Make predictions on test data
        y_pred = poly_reg_model.predict(X_test_poly)

        print('MSE polynomial_regression:', mean_squared_error(y_test, y_pred))
        save_confusion_matrix(
            y_test, np.round(y_pred).astype(int), [0, 1], "out/PolynominalRegression/confusion_matrix.png")
