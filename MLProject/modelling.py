import dagshub
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import random
import numpy as np

mlflow.set_tracking_uri("http://127.0.0.1:5000/")

mlflow.set_experiment("Membangun-Model-Klasifikasi-Diabetes")

# path dataset
train_data = pd.read_csv("pima_diabetes_processed/train_diabetes_processed.csv")
test_data = pd.read_csv("pima_diabetes_processed/test_diabetes_processed.csv")

# gunakan dataset yang sudah dibagi
X_train = train_data.drop("Outcome", axis=1)
y_train = train_data["Outcome"]
X_test = test_data.drop("Outcome", axis=1)
y_test = test_data["Outcome"]

input_example = X_train[0:5]

# aktifkan autolog sebelum proses mlflow.start_run()
mlflow.sklearn.autolog() 

with mlflow.start_run(run_name="manual_run"):    
    # Train model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)