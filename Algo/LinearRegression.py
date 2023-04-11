from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report
from utils import save_confusion_matrix, save_ROC
import numpy as np
import matplotlib.pyplot as plt


class linear_regression:
    def __init__(self, X_train, X_test, y_train, y_test, rs) -> None:

        # Create linear regression object
        reg = LinearRegression()

        # Fit the model using training data
        reg.fit(X_train, y_train)

        # Make predictions on test data
        y_pred = reg.predict(X_test)

        threshold = 0.5
        y_pred_binary = np.where(y_pred >= threshold, 1, 0)

        best_threshold = save_ROC('Linear Regression',
                                  "out/LinearRegression/ROC.png", y_test, y_pred)

        y_pred_binary = np.where(y_pred >= best_threshold, 1, 0)

        save_confusion_matrix(y_test, np.round(y_pred_binary).astype(
            int), [0, 1], "out/LinearRegression/confusion_matrix.png", 'Linear Regression')
