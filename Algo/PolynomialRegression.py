from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import classification_report
from sklearn.linear_model import LinearRegression
from utils import save_confusion_matrix, save_ROC
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

        best_threshold = save_ROC('Polynomial Regression',
                                  'out/PolynomialRegression/ROC.png', y_test, y_pred)

        y_pred_binary = np.where(y_pred >= best_threshold, 1, 0)

        save_confusion_matrix(y_test, np.round(y_pred_binary).astype(
            int), [0, 1], "out/PolynomialRegression/confusion_matrix.png")
