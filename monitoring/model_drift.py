
def check_drift(old_acc, new_acc):
    drift = abs(old_acc - new_acc)
    return drift

print(check_drift(0.95, 0.80))