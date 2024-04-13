# code for general evaluation

import numpy as np
import evaluate
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def evaluate_classification(preds, gold):
    metric = evaluate.load("bstrai/classification_report")
    return metric.compute(predictions=preds, references=gold)

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


def eval_binary(y_true, y_pred, pos_label=1, average="binary"):
    """pos_label: postive label is machine text here, label is 1, human text is 0"""
    precision, recall, F1, support = precision_recall_fscore_support(
        y_true, y_pred, pos_label = pos_label, average = average)
    # accuracy
    accuracy = accuracy_score(y_true, y_pred)
    # precison
    # pre = precision_score(y_true, y_pred, pos_label = pos_label, average = average)
    # recall
    # rec = recall_score(y_true, y_pred, pos_label = pos_label, average = average)
    metrics = {
        "accuracy": round(accuracy, 3),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "F1": round(F1, 3),
    }
    return metrics

