from DataLoader import DataLoader
from LinearRegression import linear_regression
from LogisticRegression import logistic_regression
from PolynomialRegression import polynominal_regression
from RandomForest import random_forest
from DecisionTree import desicion_tree
from NeuralNetwork import neural_network
import pandas as pd
import numpy as np


if __name__ == "__main__":
    data = DataLoader("DiabetsPredict.csv")
    X, y = data.get_data()
    X_train, X_test, y_train, y_test, rs = data.get_train_test_norm(X, y)
    linear_regression(X_train, X_test, y_train, y_test, rs)
    logistic_regression(X_train, X_test, y_train, y_test, rs)
    polynominal_regression(X_train, X_test, y_train, y_test, rs)
    desicion_tree(X_train, X_test, y_train, y_test, rs)
    random_forest(X_train, X_test, y_train, y_test, rs)
    neural_network(X_train, X_test, y_train, y_test, rs)
    pass
