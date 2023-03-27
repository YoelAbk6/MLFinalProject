from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


class logistic_regression:

    def __init__(self, X, y) -> None:
        # Assume data is stored in X (8 features) and y (2 classes)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create logistic regression object
        reg = LogisticRegression(max_iter=200)

        # Fit the model using training data
        reg.fit(X_train, y_train)

        # Make predictions on test data
        y_pred = reg.predict(X_test)

        print("Logistic Regression")

        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))


