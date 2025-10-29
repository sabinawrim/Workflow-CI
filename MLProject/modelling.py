# import dagshub
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# mlflow.set_tracking_uri("http://127.0.0.1:5000/")

# mlflow.set_experiment("Membangun-Model-Klasifikasi-Diabetes")

# path dataset
train_data = pd.read_csv("pima_diabetes_processed/train_diabetes_processed.csv")
test_data = pd.read_csv("pima_diabetes_processed/test_diabetes_processed.csv")

# gunakan dataset yang sudah dibagi
X_train = train_data.drop("Outcome", axis=1)
y_train = train_data["Outcome"]
X_test = test_data.drop("Outcome", axis=1)
y_test = test_data["Outcome"]

input_example = X_train[0:5]

with mlflow.start_run(run_name="manual_run"):
    # Log parameters
    n_estimators = 505
    max_depth = 37
    mlflow.autolog()
    
    # Train model
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example
    )
    model.fit(X_train, y_train)

    # Log metrics
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    
