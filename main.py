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
    linear_regression(X, y)
    logistic_regression(X, y)
    polynominal_regression(X, y)
    desicion_tree(X, y)
    random_forest(X, y)
    neural_network(X, y)
    pass
