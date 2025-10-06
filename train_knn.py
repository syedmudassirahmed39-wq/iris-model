import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib
import json
import pandas as pd

# Optional: Load CSV if you want
# data = pd.read_csv("data/iris.csv")
# X = data[['sepal_length','sepal_width','petal_length','petal_width']]
# y = data['species']

# Using sklearn built-in iris dataset
iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

with mlflow.start_run():
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)

    # Log parameters and metrics
    mlflow.log_param("n_neighbors", 3)
    mlflow.log_metric("accuracy", accuracy)

    # Log model to MLflow
    mlflow.sklearn.log_model(model, "knn_model")

    # Save model locally for DVC
    joblib.dump(model, "knn_model.pkl")

    # Save metrics for DVC
    with open("metrics.json", "w") as f:
        json.dump({"accuracy": accuracy}, f)

print("Training complete. Accuracy:", accuracy)
