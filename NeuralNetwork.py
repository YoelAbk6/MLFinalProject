from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix


class neural_network:
    def __init__(self, X, y) -> None:
        # Assume data is stored in X (8 features) and y (2 classes)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # create an MLPClassifier with 1 hidden layer with 10 neurons
        nn = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000)

        # fit the model to the training data
        nn.fit(X_train, y_train)

        # evaluate the model on the test data
        y_pred = nn.predict(X_test)

        print("Neural Network")
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))
