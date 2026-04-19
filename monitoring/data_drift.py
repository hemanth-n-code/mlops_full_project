# monitoring/data_drift.py

import pandas as pd
from scipy.stats import ks_2samp

# Load data
ref = pd.read_csv("data/raw/creditcard.csv").sample(2000, random_state=42)
curr = pd.read_csv("data/raw/creditcard.csv").sample(2000, random_state=24)

# Simulate drift (IMPORTANT for demo)
curr["Amount"] = curr["Amount"] * 2

drift_results = []

for col in ref.columns:
    stat, p_value = ks_2samp(ref[col], curr[col])
    drift = p_value < 0.05
    drift_results.append((col, p_value, drift))

# Save simple HTML report
html = "<h1>Data Drift Report</h1><table border=1><tr><th>Column</th><th>p-value</th><th>Drift</th></tr>"

for col, p, d in drift_results:
    html += f"<tr><td>{col}</td><td>{p:.5f}</td><td>{d}</td></tr>"

html += "</table>"

with open("monitoring/drift_report.html", "w") as f:
    f.write(html)

print("Drift report generated successfully!")