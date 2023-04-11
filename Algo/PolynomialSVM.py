from sklearn.svm import SVC
from sklearn.preprocessing import PolynomialFeatures
from utils import save_confusion_matrix, save_ROC
import numpy as np


class PolynomialSVM:
    def __init__(self, X_train, X_test, y_train, y_test, rs):
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)

        # Fit the model using training data
        svm_model = SVC(kernel='poly', degree=2)
        svm_model.fit(X_train_poly, y_train)

        # Make predictions on test data
        y_pred = svm_model.predict(X_test_poly)

        best_threshold = save_ROC('Polynomial SVM',
                                  'out/PolynomialSVM/ROC.png', y_test, y_pred)

        y_pred_binary = np.where(y_pred >= best_threshold, 1, 0)

        save_confusion_matrix(y_test, np.round(y_pred_binary).astype(
            int), [0, 1], "out/PolynomialSVM/confusion_matrix.png", 'SVM')
