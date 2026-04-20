# monitoring/model_drift.py

def detect_model_drift(old_acc, new_acc, threshold=0.05):
    drop = old_acc - new_acc

    if drop > threshold:
        print("Model drift detected")
        return True
    else:
        print("No model drift")
        return False