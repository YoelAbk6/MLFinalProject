from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.utils import check_random_state

class polynominal_regression:
    def __init__(self, X, y) -> None:
        rs = check_random_state(42)
        poly = PolynomialFeatures(degree=2, include_bias=False)

        poly_features = poly.fit_transform(X)

        # Assume data is stored in X (8 features) and y (2 classes)
        X_train, X_test, y_train, y_test = train_test_split(poly_features, y, test_size=0.2, random_state=rs)

        # Create linear regression object
        poly_reg_model = LinearRegression()

        # Fit the model using training data
        poly_reg_model.fit(X_train, y_train)

        # Make predictions on test data
        y_pred = poly_reg_model.predict(X_test)

        print('MSE polynomial_regression:', mean_squared_error(y_test, y_pred))
