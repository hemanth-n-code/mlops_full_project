# src/train.py

import sys
import os
import pickle
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_preprocessing import load_data
from monitoring.data_drift import detect_data_drift
from monitoring.model_drift import detect_model_drift

# -----------------------------
# CONFIG
# -----------------------------
MODEL_NAME = "fraud-model"

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
with mlflow.start_run() as run:

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    new_acc = model.score(X_test, y_test)
    print("New model accuracy:", new_acc)

    # -----------------------------
    # LOG TO MLFLOW
    # -----------------------------
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy_new", new_acc)
    mlflow.log_metric("accuracy_old", old_acc)
    mlflow.log_metric("data_drift", int(drift_detected))

    # -----------------------------
    # MODEL DRIFT CHECK
    # -----------------------------
    model_drift = detect_model_drift(old_acc, new_acc)

    # -----------------------------
    # MODEL SELECTION
    # -----------------------------
    if new_acc > old_acc:
        print("✅ New model better → Promote to STAGING")

        os.makedirs("models", exist_ok=True)
        with open("models/model.pkl", "wb") as f:
            pickle.dump(model, f)

        # 🔥 Register model in MLflow
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name=MODEL_NAME
        )

        # Get latest version and move to STAGING
        client = mlflow.tracking.MlflowClient()

        latest_version = client.get_latest_versions(MODEL_NAME, stages=["None"])[0].version

        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=latest_version,
            stage="Staging"
        )

    else:
        print("❌ Old model retained → stays in PRODUCTION")

print("Pipeline complete")