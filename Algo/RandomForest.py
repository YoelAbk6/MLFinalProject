from sklearn.ensemble import RandomForestClassifier
from utils import save_confusion_matrix, save_ROC, print_percent


class random_forest:

    def __init__(self, X_train, X_test, y_train, y_test, rs, out_folder, is_best_threshold = False) -> None:

        # Create a Random Forest classifier with 100 trees
        rf_model = RandomForestClassifier(n_estimators=100, random_state=rs)

        # Train the Random Forest model
        rf_model.fit(X_train, y_train)

        # Make predictions on the test data
        y_pred = rf_model.predict(X_test)

        y_prob = rf_model.predict_proba(X_test)[:, 1]

        best_threshold = save_ROC(
            'Random Forest', f"{out_folder}/RandomForest/ROC.png", y_test, y_prob)
        
        if is_best_threshold:
            y_pred = (y_prob >= best_threshold).astype(int)

        save_confusion_matrix(
            y_test, y_pred, [0, 1], f"{out_folder}/RandomForest/confusion_matrix.png", 'Random Forest')

        print_percent(y_test, y_pred, "Random Forest")
