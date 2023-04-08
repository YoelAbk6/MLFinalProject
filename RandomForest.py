from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import check_random_state
from utils import save_confusion_matrix


class random_forest:

    def __init__(self, X_train, X_test, y_train, y_test, rs) -> None:

        # Create a Random Forest classifier with 100 trees
        rf_model = RandomForestClassifier(n_estimators=100, random_state=rs)

        # Train the Random Forest model
        rf_model.fit(X_train, y_train)

        # Make predictions on the test data
        y_pred = rf_model.predict(X_test)

        print("Random Forest")

        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        save_confusion_matrix(
            y_test, y_pred, [0, 1], "out/RandomForest/confusion_matrix.png")
