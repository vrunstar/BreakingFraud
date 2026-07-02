import numpy as np
from sklearn.metrics import confusion_matrix

def find_optimal_threshold(y_test, y_proba, fn_cost=25000, fp_cost=25):
    thresholds = np.arange(0.01, 1.0, 0.01)
    costs = []

    for t in thresholds:
        y_pred_t = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_t).ravel()
        total_cost = (fn * fn_cost) + (fp * fp_cost)
        costs.append(total_cost)

    best_idx = np.argmin(costs)
    best_threshold = thresholds[best_idx]
    best_cost = costs[best_idx]

    return best_threshold, best_cost, thresholds, costs