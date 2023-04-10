from DataLoader import DataLoader
from Algo.LinearRegression import linear_regression
from Algo.LogisticRegression import logistic_regression
from Algo.PolynomialRegression import polynominal_regression
from Algo.RandomForest import random_forest
from Algo.DecisionTree import decision_tree
from Algo.NeuralNetwork import neural_network


if __name__ == "__main__":
    data = DataLoader("DiabetsPredict.csv")
    X, y = data.get_clean_data()
    X_train, X_test, y_train, y_test, rs = data.get_train_test_norm(X, y)
    linear_regression(X_train, X_test, y_train, y_test, rs)
    logistic_regression(X_train, X_test, y_train, y_test, rs)
    polynominal_regression(X_train, X_test, y_train, y_test, rs)
    decision_tree(X_train, X_test, y_train, y_test, rs)
    random_forest(X_train, X_test, y_train, y_test, rs)
    neural_network(X_train, X_test, y_train, y_test, rs)
    pass
