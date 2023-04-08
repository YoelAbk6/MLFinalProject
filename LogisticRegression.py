from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import check_random_state
from utils import save_confusion_matrix


class logistic_regression:

    def __init__(self, X_train, X_test, y_train, y_test, rs) -> None:

        # Create logistic regression object
        reg = LogisticRegression(max_iter=200, random_state=rs)

        # Fit the model using training data
        reg.fit(X_train, y_train)

        # Make predictions on test data
        y_pred = reg.predict(X_test)

        print("Logistic Regression")

        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        save_confusion_matrix(
            y_test, y_pred, [0, 1], "out/LogisticRegression/confusion_matrix.png")
