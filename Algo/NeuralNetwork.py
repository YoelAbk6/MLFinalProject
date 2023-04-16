from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from utils import save_confusion_matrix, save_ROC, print_percent
import matplotlib.pyplot as plt


class neural_network:
    def __init__(self, X_train, X_test, y_train, y_test, rs, out_folder, is_best_threshold=False) -> None:
        # create an MLPClassifier with 1 hidden layer with 10 neurons
        nn = MLPClassifier(hidden_layer_sizes=(
            10,), max_iter=1000, early_stopping=True, n_iter_no_change=800, random_state=rs)

        # fit the model to the training data
        nn.fit(X_train, y_train)

        # evaluate the model on the test data
        y_pred = nn.predict(X_test)

        y_prob = nn.predict_proba(X_test)[:, 1]

        best_threshold = save_ROC(
            'Neural Network', f"{out_folder}/NeuralNetwork/ROC.png", y_test, y_prob)

        if is_best_threshold:
            y_pred = (y_prob >= best_threshold).astype(int)

        save_confusion_matrix(
            y_test, y_pred, [0, 1], f"{out_folder}/NeuralNetwork/confusion_matrix.png", 'Neural Network')
        self.save_accuracy_and_loss_graphs(
            nn.validation_scores_, nn.loss_curve_, out_folder)

        print_percent(y_test, y_pred, "Neural Network")

        report = classification_report(y_test, y_pred)
        print(report)

    def save_accuracy_and_loss_graphs(self, accuracy, loss, out_folder) -> None:
        plt.clf()
        plt.plot(accuracy)
        plt.title("Accuracy")
        plt.savefig(f"{out_folder}/NeuralNetwork/Accuracy.png")

        plt.clf()
        plt.plot(loss)
        plt.title("Loss")
        plt.savefig(f"{out_folder}/NeuralNetwork/Loss.png")
