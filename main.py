from DataLoader import DataLoader
from  LinearRegression import linear_regression
import LogisticRegression
import PolynomialRegression
import RandomForest
import DecisionTree
import NeuralNetwork
import pandas as pd
import numpy as np


if __name__ == "__main__":
    data = DataLoader("DiabetsPredict.csv")
    X, y = data.get_data()
    linear_regression(X, y)
    pass
