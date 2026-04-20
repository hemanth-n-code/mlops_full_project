import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from data_preprocessing import load_data
import pickle
import os

# ✅ Check if dataset exists
if not os.path.exists("data/raw/creditcard.csv"):
    print("Dataset not found → skipping training (CI safe)")
    exit()

X, y = load_data()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

mlflow.set_experiment("fraud-detection")

with mlflow.start_run():

    model = RandomForestClassifier(n_estimators=50)
    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)

    mlflow.log_param("n_estimators", 50)
    mlflow.log_metric("accuracy", acc)

    os.makedirs("models", exist_ok=True)
    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)

    mlflow.sklearn.log_model(model, "model")

print("Training complete")