import numpy as np

def calculate_metrics(results, y) -> tuple[float, float, float]:
    if not isinstance(results, np.ndarray):
        results = np.array(results)
    if not isinstance(y, np.ndarray):
        y = np.array(y)

    TP = np.sum(results & (y == 1))
    FP = np.sum(results & (y == 0))
    FN = np.sum((results == False) & (y == 1))
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    accuracy = (results == y).mean()
    return accuracy, precision, recall