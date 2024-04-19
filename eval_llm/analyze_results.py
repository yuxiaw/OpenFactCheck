import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import argparse


def read_eval_result(intermediate_results_dir, datasets):
    # Read from the JSON file
    with open(
            os.path.join(intermediate_results_dir, "combined_result.json"), "r"
    ) as json_file:
        combined_result = json.load(json_file)

    # Access the dictionaries
    for idx, result in enumerate(combined_result):
        print(f"Dataset {datasets[idx]}:")
        for key, value in result.items():
            print(f"{key}: {value}")
    return combined_result


def barplot_free_response(combined_result, fig_path):
    # Given dictionary
    data = combined_result[-1]
    del data['Percentage of true responses']

    # Custom x-axis tick labels
    xticks = ["factool-qa", "felm-wk", "factcheckgpt"]

    # Define colors for each subplot
    colors = ["skyblue", "lightgreen", "salmon"]

    # Create subplots
    fig, axs = plt.subplots(1, len(data), figsize=(15, 5))

    # Plot each key-value pair in a subplot with corresponding color
    for i, (key, values) in enumerate(data.items()):
        bars = axs[i].bar(
            range(len(values)), values, color=colors[i]
        )  # Set color for each subplot
        axs[i].set_title(key)
        axs[i].set_xticks(range(len(values)))  # Set xticks
        axs[i].set_xticklabels(xticks)  # Set custom tick labels

        # Add numbers on top of the bars
        for bar in bars:
            height = bar.get_height()
            axs[i].text(
                bar.get_x() + bar.get_width() / 2,
                height,
                round(height, 2),
                ha="center",
                va="bottom",
            )

    # Set figure title
    fig.suptitle("Free-style Response Evaluation")

    # Show plot
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(fig_path, "free_style_response_barchart.pdf"), format="pdf")
    plt.savefig(os.path.join(fig_path, "free_style_response_barchart.png"), format="png")


def piechart_freshqa(combined_result, fig_path):
    # Given numbers
    result = combined_result[2]
    sizes = [result["Accuracy"], 1 - result["Accuracy"]]
    labels = ["True Answer", "False Answer"]
    colors = [(0, 1, 0, 0.5), (1, 0, 0, 0.5)]  # Red and green with 50% transparency

    # Plot pie chart
    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=140)
    plt.title("True vs False Answers on FreshQA")

    # Equal aspect ratio ensures that pie is drawn as a circle
    plt.axis("equal")

    # Show plot
    # plt.show()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, "freshqa_piechart.pdf"), format="pdf")
    plt.savefig(os.path.join(fig_path, "freshqa_piechart.png"), format="png")


def piechart_freshqa(combined_result, fig_path):
    # Given numbers
    result = combined_result[2]
    sizes = [result["Accuracy"], 1 - result["Accuracy"]]
    labels = ["True Answer", "False Answer"]
    colors = [(0, 1, 0, 0.5), (1, 0, 0, 0.5)]  # Red and green with 50% transparency

    # Plot pie chart
    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=140)
    plt.title("True vs False Answers on FreshQA")

    # Equal aspect ratio ensures that pie is drawn as a circle
    plt.axis("equal")

    # Show plot
    # plt.show()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, "freshqa_piechart.pdf"), format="pdf")
    plt.savefig(os.path.join(fig_path, "freshqa_piechart.png"), format="png")


def barplot_selfaware(combined_result, fig_path):
    # Data
    result = combined_result[1]
    unanswerable_as_pos = result["unanswerable_as_pos"]
    answerable_as_pos = result["answerable_as_pos"]

    metrics = list(unanswerable_as_pos.keys())
    unanswerable_values = list(unanswerable_as_pos.values())
    answerable_values = list(answerable_as_pos.values())

    # Plotting the bar chart
    width = 0.35  # Width of the bars
    x = range(len(metrics))
    plt.figure(figsize=(6, 6))
    fig, ax = plt.subplots()
    rects1 = ax.bar(x, unanswerable_values, width, label="Unanswerable_as_positive")
    rects2 = ax.bar(
        [i + width for i in x], answerable_values, width, label="Answerable_as_positive"
    )

    # Adding values on top of each bar
    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.annotate(
            "{}".format(height),
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    # Adding labels and title
    ax.set_xlabel("Metrics")
    ax.set_ylabel("Performance")
    ax.set_title("Performance on SelfAware")
    ax.set_xticks([i + width / 2 for i in x])
    ax.set_xticklabels(metrics)
    ax.set_ylim((0, max(unanswerable_values + answerable_values) + 0.25))
    ax.legend()

    # Displaying the plot
    # plt.show()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, "selfaware_performance.pdf"), format="pdf")
    plt.savefig(os.path.join(fig_path, "selfaware_performance.png"), format="png")


def cm_selfaware(intermediate_results_dir, response_column_name, fig_path):
    # gold labels vs predictions
    df = pd.read_json(
        f"{intermediate_results_dir}/selfaware_{response_column_name}.json"
    )
    y_true = df["gold_labels"].to_list()
    y_pred = df["predictions"].to_list()

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    # Plot confusion matrix
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix on SelfAware")
    plt.colorbar()
    classes = ["Answerable", "Unanswerable"]
    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    # plt.show()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, "selfaware_cm.pdf"), format="pdf")
    plt.savefig(os.path.join(fig_path, "selfaware_cm.png"), format="png")


def barplot_snowballing_acc(combined_result, fig_path):
    # Data
    result = combined_result[0]
    items = result.keys()
    values = [v["accuracy"] for k, v in result.items()]
    colors = ["red", "blue", "green", "orange"]
    plt.figure(figsize=(6, 6))
    # Plotting the bar chart
    bars = plt.bar(items, values, color=colors, alpha=0.4, width=0.5)

    # Adding values on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            str(height),
            ha="center",
            va="bottom",
        )
    # Rotating x-axis tick labels
    plt.xticks(rotation=20)  # Rotate labels by 45 degrees

    # Adding labels and title
    plt.xlabel("Topics")
    plt.ylabel("Accuracy")
    plt.title("Accuracy on Snowballing dataset.")
    plt.ylim((0, max(values) + 0.1))

    # Displaying the plot
    # plt.show()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, "snowballing_acc.pdf"), format="pdf")
    plt.savefig(os.path.join(fig_path, "snowballing_acc.png"), format="png")


def compute_overall_score(combined_result, intermediate_results_dir):
    score = sum([
        combined_result[0]['All']['accuracy'],
        (combined_result[1]['answerable_as_pos']['accuracy'] + combined_result[1]['unanswerable_as_pos'][
            'accuracy']) / 2,
        combined_result[2]['Accuracy'],
        sum(combined_result[5]['Percentage of true responses'])
    ]) / 6

    json.dump({"overall": score}, open(os.path.join(intermediate_results_dir, "overall_score.json"), 'w'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis_result_path", default="./fig")
    parser.add_argument("--input_path", default="./intermediate_results")
    parser.add_argument(
        "--model_response_path", default="./model_response/GPT4_responses.csv"
    )
    args = parser.parse_args()

    intermediate_results_dir = args.input_path
    test_responses_dir = args.model_response_path
    fig_path = args.analysis_result_path
    os.makedirs(fig_path, exist_ok=True)

    datasets = [
        "snowballing",
        "selfaware",
        "freshqa",
        "factool-qa",
        "felm-wk",
        "factcheckgpt",
    ]

    responses_to_test = pd.read_csv(test_responses_dir)
    response_column_name = responses_to_test.columns.tolist()[-1]

    combined_result = read_eval_result(intermediate_results_dir, datasets)
    compute_overall_score(combined_result, intermediate_results_dir)
    # performance over each dataset
    barplot_snowballing_acc(combined_result, fig_path)
    cm_selfaware(intermediate_results_dir, response_column_name, fig_path)
    barplot_selfaware(combined_result, fig_path)
    piechart_freshqa(combined_result, fig_path)
    barplot_free_response(combined_result, fig_path)

    # performance over topics
    # to-do in the future


if __name__ == "__main__":
    main()
