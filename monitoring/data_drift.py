# monitoring/data_drift.py

import pandas as pd
from scipy.stats import ks_2samp

def detect_data_drift():

    ref = pd.read_csv("data/raw/creditcard.csv").sample(2000, random_state=42)
    curr = pd.read_csv("data/raw/creditcard.csv").sample(2000, random_state=24)

    # simulate drift
    curr["Amount"] = curr["Amount"] * 2

    drift_results = []
    drift_count = 0

    for col in ref.columns:
        stat, p_value = ks_2samp(ref[col], curr[col])
        drift = p_value < 0.05

        if drift:
            drift_count += 1

        drift_results.append((col, p_value, drift))

    drift_detected = drift_count > 3  # threshold

    print(f"Drifted columns: {drift_count}")

    return drift_detected