from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

class random_forest:

    def __init__(self, X, y) -> None:
        # Assume data is stored in X (8 features) and y (2 classes)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create a Random Forest classifier with 100 trees
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

        # Train the Random Forest model
        rf_model.fit(X_train, y_train)

        # Make predictions on the test data
        y_pred = rf_model.predict(X_test)

        print("Random Forest")

        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))
