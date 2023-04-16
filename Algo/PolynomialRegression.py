from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report
from utils import save_confusion_matrix, save_ROC, print_percent
import numpy as np


class polynominal_regression:
    def __init__(self, X_train, X_test, y_train, y_test, rs, out_folder, is_best_threshold=False) -> None:
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)

        # Fit the model using training data
        poly_reg_model = LinearRegression()
        poly_reg_model.fit(X_train_poly, y_train)

        # Make predictions on test data
        y_pred = poly_reg_model.predict(X_test_poly)

        threshold = 0.5
        y_pred = np.where(y_pred >= threshold, 1, 0)

        if not is_best_threshold:
            y_pred = np.round(y_pred).astype(int)

        best_threshold = save_ROC('Polynomial Regression',
                                  f"{out_folder}/PolynomialRegression/ROC.png", y_test, y_pred)

        if is_best_threshold:
            y_pred = np.where(y_pred >= best_threshold, 1, 0)

        save_confusion_matrix(y_test, np.round(y_pred).astype(
            int), [0, 1], f"{out_folder}/PolynomialRegression/confusion_matrix.png", 'Polynomial Regression')

        print_percent(y_test, y_pred, "Polynomial Regression")

        report = classification_report(y_test, y_pred)
        print(report)
