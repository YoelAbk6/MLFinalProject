from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import classification_report

class desicion_tree:

    def __init__(self, X, y) -> None:
        # Assume data is stored in X (8 features) and y (2 classes)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create decision tree classifier object
        clf = DecisionTreeClassifier()

        # Train the classifier on the training data
        clf.fit(X_train, y_train)

        # Make predictions on the testing data
        y_pred = clf.predict(X_test)

        print("Decision Tree")

        confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
        print("confusion matrix:")
        print(confusion_matrix)
        print(classification_report(y_test, y_pred))
