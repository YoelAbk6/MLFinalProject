from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.utils import check_random_state
from utils import save_confusion_matrix


class linear_regression:
    def __init__(self, X, y) -> None:
        rs = check_random_state(42)
        # Assume data is stored in X (8 features) and y (2 classes)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=rs)

        # Create linear regression object
        reg = LinearRegression()

        # Fit the model using training data
        reg.fit(X_train, y_train)

        # Make predictions on test data
        y_pred = reg.predict(X_test)

        print('MSE linear_regression:', mean_squared_error(y_test, y_pred))
        save_confusion_matrix(
            y_test, y_pred, [0, 1], "out/LinearRegression/confusion_matrix.png")
