from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import check_random_state
from utils import save_confusion_matrix


class neural_network:
    def __init__(self, X_train, X_test, y_train, y_test, rs) -> None:

        # create an MLPClassifier with 1 hidden layer with 10 neurons
        nn = MLPClassifier(hidden_layer_sizes=(
            10,), max_iter=1000, random_state=rs)

        # fit the model to the training data
        nn.fit(X_train, y_train)

        # evaluate the model on the test data
        y_pred = nn.predict(X_test)

        print("Neural Network")
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        save_confusion_matrix(
            y_test, y_pred, [0, 1], "out/NeuralNetwork/confusion_matrix.png")
