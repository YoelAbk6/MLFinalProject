from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.utils import check_random_state
from utils import save_confusion_matrix
import numpy as np


class linear_regression:
    def __init__(self, X_train, X_test, y_train, y_test, rs) -> None:

        # Create linear regression object
        reg = LinearRegression()

        # Fit the model using training data
        reg.fit(X_train, y_train)

        # Make predictions on test data
        y_pred = reg.predict(X_test)

        print('MSE linear_regression:', mean_squared_error(y_test, y_pred))
        save_confusion_matrix(
            y_test, np.round(y_pred).astype(int), [0, 1], "out/LinearRegression/confusion_matrix.png")
