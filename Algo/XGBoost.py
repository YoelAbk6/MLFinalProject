import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from utils import save_confusion_matrix, save_ROC, print_percent


class XGBoost:
    def __init__(self, X_train, X_test, y_train, y_test, rs, out_folder, is_best_threshold=False) -> None:

        # define XGBoost model
        model = XGBClassifier()

        # train XGBoost model
        model.fit(X_train, y_train)

        # make predictions on test data
        y_pred = model.predict(X_test)

        threshold = 0.5
        y_pred_binary = np.where(y_pred >= threshold, 1, 0)

        best_threshold = save_ROC('XGBoost',
                                  f"{out_folder}/XGBoost/ROC.png", y_test, y_pred)
        if is_best_threshold:
            y_pred_binary = np.where(y_pred >= best_threshold, 1, 0)

        save_confusion_matrix(y_test, np.round(y_pred_binary).astype(
            int), [0, 1], f"{out_folder}/XGBoost/confusion_matrix.png", 'XGBoost')

        print_percent(y_test, y_pred, "XGBoost")

        report = classification_report(y_test, y_pred)
        print(report)
