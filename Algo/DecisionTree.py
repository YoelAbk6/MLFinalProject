from sklearn.tree import DecisionTreeClassifier
from utils import save_confusion_matrix, save_ROC, print_percent


class decision_tree:

    def __init__(self, X_train, X_test, y_train, y_test, rs, out_folder, is_best_threshold=False) -> None:
        # Create decision tree classifier object
        clf = DecisionTreeClassifier(random_state=rs)

        # Train the classifier on the training data
        clf.fit(X_train, y_train)

        # Make predictions on the testing data
        y_pred = clf.predict(X_test)

        y_prob = clf.predict_proba(X_test)[:, 1]

        best_threshold = save_ROC(
            "Decision Tree", f"{out_folder}/DecisionTree/ROC.png", y_test, y_prob)

        if is_best_threshold:
            y_pred = (y_prob >= best_threshold).astype(int)

        save_confusion_matrix(
            y_test, y_pred, [0, 1], f"{out_folder}/DecisionTree/confusion_matrix.png", 'Decision Tree')

        print_percent(y_test, y_pred, "Decision Tree")
