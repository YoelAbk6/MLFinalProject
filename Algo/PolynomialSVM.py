from sklearn.svm import SVC
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import classification_report
from utils import save_confusion_matrix, save_ROC, print_percent
import numpy as np


class PolynomialSVM:
    def __init__(self, X_train, X_test, y_train, y_test, rs, out_folder, is_best_threshold=False):
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)

        # Fit the model using training data
        svm_model = SVC(kernel='poly', degree=2)
        svm_model.fit(X_train_poly, y_train)

        # Make predictions on test data
        y_pred = svm_model.predict(X_test_poly)

        best_threshold = save_ROC('Polynomial SVM',
                                  f"{out_folder}/PolynomialSVM/ROC.png", y_test, y_pred)

        if is_best_threshold:
            y_pred = np.where(y_pred >= best_threshold, 1, 0)

        save_confusion_matrix(y_test, np.round(y_pred).astype(
            int), [0, 1], f"{out_folder}/PolynomialSVM/confusion_matrix.png", 'SVM')

        print_percent(y_test, y_pred, "PolynomialSVM")

        report = classification_report(y_test, y_pred)
        print(report)
