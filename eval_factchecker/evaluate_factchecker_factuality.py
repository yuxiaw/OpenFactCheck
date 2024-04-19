import json
import argparse
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    f1_score,
    recall_score,
)

def eval_binary_classification(y_true, y_pred, pos_label="yes"):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=pos_label)
    recall = recall_score(y_true, y_pred, pos_label=pos_label)
    F1 = f1_score(y_true, y_pred, pos_label=pos_label)

    metrics = {
        "accuracy": round(accuracy, 3),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "F1": round(F1, 3),
    }
    return metrics


def evaluate_checker_performance(df_gold, df_checker):
    assert len(df_checker.columns) == 3
    gold_labels = df_gold['claim_label'].to_list()
    predictions = df_checker[df_checker.columns[0]].to_list()

    assert (len(gold_labels) == len(predictions))
    for i, (g, p) in enumerate(zip(gold_labels, predictions)):
        if isinstance(g, bool) and isinstance(p, bool):
            pass
        else:
            print(i, type(p), type(g))

    # evalaute performance
    r1 = eval_binary_classification(y_true=gold_labels, y_pred=predictions, pos_label=True)
    r2 = eval_binary_classification(y_true=gold_labels, y_pred=predictions, pos_label=False)

    total_time = 0
    total_cost = 0
    if "time" in df_checker.columns[1]:
        total_time = df_checker[df_checker.columns[1]].astype(float).sum()
    if "cost" in df_checker.columns[2]:
        total_cost = df_checker[df_checker.columns[2]].astype(float).sum()

    print(f"True as postive label: {r1} \nFalse as positive label: {r2}")
    return {
        "True_as_positive": r1,
        "False_as_positive": r2,
        "total_time": total_time,
        "total_cost": total_cost,
        "num_samples": len(predictions)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gold_path",
        default="./data/check_claims.jsonl",
        help="Path to gold labels",
    )
    parser.add_argument(
        "--input_path",
        help="Path to the checker's results",
    )
    parser.add_argument(
        "--results_path",
        default="./data/df_checkers.csv",
        help="Path to save the results",
    )

    args = parser.parse_args()

    # load gold and predicted labels
    df_gold = pd.read_json(args.gold_path, lines=True)
    df_checker = pd.read_csv(args.input_path, index_col=None)

    results = evaluate_checker_performance(df_gold, df_checker)

    # Save the results
    with open(f"{args.results_path}/results.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()


