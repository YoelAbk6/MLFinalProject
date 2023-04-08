from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import check_random_state
from utils import save_confusion_matrix


class desicion_tree:

    def __init__(self, X_train, X_test, y_train, y_test, rs) -> None:

        # Create decision tree classifier object
        clf = DecisionTreeClassifier(random_state=rs)

        # Train the classifier on the training data
        clf.fit(X_train, y_train)

        # Make predictions on the testing data
        y_pred = clf.predict(X_test)

        print("Decision Tree")

        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        save_confusion_matrix(
            y_test, y_pred, [0, 1], "out/DesicionTree/confusion_matrix.png")
