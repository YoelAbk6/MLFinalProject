from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from utils import save_confusion_matrix, save_ROC


class random_forest:

    def __init__(self, X_train, X_test, y_train, y_test, rs) -> None:

        # Create a Random Forest classifier with 100 trees
        rf_model = RandomForestClassifier(n_estimators=100, random_state=rs)

        # Train the Random Forest model
        rf_model.fit(X_train, y_train)

        # Make predictions on the test data
        y_pred = rf_model.predict(X_test)

        y_prob = rf_model.predict_proba(X_test)[:, 1]

        best_threshold = save_ROC(
            'Random Forest', 'out/RandomForest/ROC.png', y_test, y_prob)

        y_pred = (y_prob >= best_threshold).astype(int)

        save_confusion_matrix(
            y_test, y_pred, [0, 1], "out/RandomForest/confusion_matrix.png")
