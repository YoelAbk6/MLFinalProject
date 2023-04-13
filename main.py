from DataLoader import DataLoader
from Algo.LinearRegression import linear_regression
from Algo.LogisticRegression import logistic_regression
from Algo.PolynomialRegression import polynominal_regression
from Algo.RandomForest import random_forest
from Algo.DecisionTree import decision_tree
from Algo.NeuralNetwork import neural_network
from Algo.PolynomialSVM import PolynomialSVM

if __name__ == "__main__":
    data = DataLoader("DiabetsPredict.csv")
    models = [PolynomialSVM, linear_regression, logistic_regression, polynominal_regression, decision_tree, random_forest, neural_network]

    # First run, raw data
    X, y = data.get_raw_data()
    X_train, X_test, y_train, y_test, rs = data.get_train_test_norm(X, y)
    for model in models:
        model(X_train, X_test, y_train, y_test, rs, 'raw_out')
    
    # Second run, clean data
    X, y = data.get_clean_data()
    X_train, X_test, y_train, y_test, rs = data.get_train_test_norm(X, y)
    for model in models:
        model(X_train, X_test, y_train, y_test, rs, 'clean_out')
    
    # Third run, clean data and best threshold
    X, y = data.get_clean_data()
    X_train, X_test, y_train, y_test, rs = data.get_train_test_norm(X, y)
    for model in models:
        model(X_train, X_test, y_train, y_test, rs, 'best_out', is_best_threshold = True)
