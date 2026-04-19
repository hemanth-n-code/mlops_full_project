import pandas as pd

def load_data():
    df = pd.read_csv("data/raw/creditcard.csv")

    # reduce size for faster training
    df = df.sample(5000, random_state=42)

    X = df.drop("Class", axis=1)
    y = df["Class"]

    return X, y