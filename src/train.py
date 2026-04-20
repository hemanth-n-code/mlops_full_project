# src/train.py
import sys
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from data_preprocessing import load_data
import pickle
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from monitoring.data_drift import detect_data_drift
from monitoring.model_drift import detect_model_drift

# -----------------------------
# LOAD DATA
# -----------------------------
if not os.path.exists("data/raw/creditcard.csv"):
    print("Dataset not found → skipping")
    exit()

X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

mlflow.set_experiment("fraud-detection")

# -----------------------------
# LOAD OLD MODEL (Champion)
# -----------------------------
old_acc = 0
if os.path.exists("models/model.pkl"):
    old_model = pickle.load(open("models/model.pkl", "rb"))
    old_acc = old_model.score(X_test, y_test)
    print("Old model accuracy:", old_acc)

# -----------------------------
# DATA DRIFT CHECK
# -----------------------------
drift_detected = detect_data_drift()

# -----------------------------
# TRAIN NEW MODEL (Challenger)
# -----------------------------
with mlflow.start_run():

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    new_acc = model.score(X_test, y_test)

    print("New model accuracy:", new_acc)

    # MLflow logging
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy_new", new_acc)
    mlflow.log_metric("accuracy_old", old_acc)
    mlflow.log_metric("data_drift", int(drift_detected))

    # -----------------------------
    # MODEL DRIFT CHECK
    # -----------------------------
    model_drift = detect_model_drift(old_acc, new_acc)

    # -----------------------------
    # MODEL SELECTION (Champion vs Challenger)
    # -----------------------------
    if new_acc > old_acc:
        print("New model is better → promoting to STAGING")

        os.makedirs("models", exist_ok=True)
        with open("models/model.pkl", "wb") as f:
            pickle.dump(model, f)

        mlflow.sklearn.log_model(model, "model")

        # Tag as staging
        mlflow.set_tag("stage", "staging")

    else:
        print("Old model retained")

        mlflow.set_tag("stage", "production")

print("Pipeline complete")