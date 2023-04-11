from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from utils import save_confusion_matrix, save_ROC


class decision_tree:

    def __init__(self, X_train, X_test, y_train, y_test, rs) -> None:

        # Create decision tree classifier object
        clf = DecisionTreeClassifier(random_state=rs)

        # Train the classifier on the training data
        clf.fit(X_train, y_train)

        # Make predictions on the testing data
        y_pred = clf.predict(X_test)

        y_prob = clf.predict_proba(X_test)[:, 1]

        best_threshold = save_ROC(
            "Decision Tree", "out/DecisionTree/ROC.png", y_test, y_prob)

        y_pred = (y_prob >= best_threshold).astype(int)

        save_confusion_matrix(
            y_test, y_pred, [0, 1], "out/DecisionTree/confusion_matrix.png", 'Decision Tree')
