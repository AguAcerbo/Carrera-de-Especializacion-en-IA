import numpy as np


def metric_pra(truth, pred):
    # Calculo el valor True Positive
    true_positive = sum(np.logical_and(truth, pred))

    # Calculo el valor True Negative
    true_negative = sum(np.logical_and(np.logical_not(truth), np.logical_not(pred)))

    # Calculo el valor False Negative
    false_negative = sum(np.logical_and(truth, np.logical_not(pred)))

    # Calculo el valor False Positive
    false_positive = sum(np.logical_and(np.logical_not(truth), pred))

    # Metricas
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)

    return true_positive, true_negative, false_negative, false_positive, precision, recall, accuracy


def mse(target, prediction):
    return np.sum((target - prediction) ** 2) / target.size
