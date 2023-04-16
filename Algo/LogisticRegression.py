from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from utils import save_confusion_matrix, save_ROC, print_percent


class logistic_regression:

    def __init__(self, X_train, X_test, y_train, y_test, rs, out_folder, is_best_threshold=False) -> None:
        # Create logistic regression object
        reg = LogisticRegression(max_iter=200, random_state=rs)

        # Fit the model using training data
        reg.fit(X_train, y_train)

        # Make predictions on test data
        y_pred = reg.predict(X_test)

        y_prob = reg.predict_proba(X_test)[:, 1]

        best_threshold = save_ROC("Logistic Regression",
                                  f"{out_folder}/LogisticRegression/ROC.png", y_test, y_prob)

        if is_best_threshold:
            y_pred = (y_prob >= best_threshold).astype(int)

        save_confusion_matrix(
            y_test, y_pred, [0, 1], f"{out_folder}/LogisticRegression/confusion_matrix.png", "Logistic Regression")

        print_percent(y_test, y_pred, "Logistic Regression")

        report = classification_report(y_test, y_pred)
        print(report)
