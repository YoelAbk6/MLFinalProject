from DataLoader import DataLoader
from  LinearRegression import linear_regression
from LogisticRegression import logistic_regression
import PolynomialRegression
import RandomForest
from DecisionTree import desicion_tree
import NeuralNetwork
import pandas as pd
import numpy as np


if __name__ == "__main__":
    data = DataLoader("DiabetsPredict.csv")
    X, y = data.get_data()
    linear_regression(X, y)
    logistic_regression(X, y)
    #PolynomialRegression
    desicion_tree(X,y)
    pass
