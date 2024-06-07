import random
import pandas as pd
from collections import Counter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, precision_score, f1_score, recall_score

def eval_classification(y_true, y_pred, average="macro"):
    precision, recall, F1, support = precision_recall_fscore_support(y_true, y_pred, average=average)
    accuracy = accuracy_score(y_true, y_pred)

    metrics = {
        "accuracy": round(accuracy, 3),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "F1": round(F1, 3),
    }
    return metrics


def eval_binary_classification(y_true, y_pred, pos_label="yes"):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=pos_label)
    recall = recall_score(y_true, y_pred, pos_label=pos_label)
    F1 = f1_score(y_true, y_pred, pos_label=pos_label)

    metrics = {
        "accuracy": round(accuracy, 2),
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "F1": round(F1, 2),
    }
    return metrics