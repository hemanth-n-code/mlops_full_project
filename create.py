import pandas as pd

df = pd.read_csv("data/raw/creditcard.csv")

fraud_row = df[df["Class"] == 1].iloc[0]
print(fraud_row)