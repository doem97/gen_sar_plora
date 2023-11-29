import json
# import glob
import os
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    precision_recall_fscore_support,
)
import pandas as pd
import argparse
import numpy as np
import datetime


def main(pred_dir, ann="./data/fusrs_v2/meta/test.txt", log_dir="./log.txt"):
    # Load predicted labels from pred.json
    # Find the only .json file starting with "best_f1_score" in pred_dir
    # json_files = glob.glob(os.path.join(pred_dir, "best_f1_score*.json"))
    # if len(json_files) == 1:
    #     pred = json_files[0]
    # else:
    #     raise ValueError(
    #         "Expected exactly one best_f1_score JSON file in pred_dir, but found {}.".format(
    #             len(json_files)
    #         )
    #     )
    if not os.path.exists(pred_dir):
        raise ValueError(f"{pred_dir} does not exist")

    with open(pred_dir, "r") as f:
        pred_data = json.load(f)
        pred_labels = pred_data["pred_label"]

    # Load ground truth labels from label.txt
    gt_labels = []
    with open(ann, "r") as f:
        for line in f:
            parts = line.strip().split()
            gt_labels.append(int(parts[-1]))

    # Calculate the per-category precision, recall, and F1-score
    precision, recall, f1, _ = precision_recall_fscore_support(
        gt_labels, pred_labels, average=None
    )

    with open(log_dir, "a") as log_f:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"************** {current_time} **************", file=log_f)
        print(f"prediction json file: {pred_dir}", file=log_f)
        # Calculate unweighted average (Macro) of FPR
        # Calculate macro F1-score
        macro_f1 = f1_score(gt_labels, pred_labels, average="macro")
        macro_pre = precision_score(gt_labels, pred_labels, average="macro")
        macro_rec = recall_score(gt_labels, pred_labels, average="macro")
        # Print the results for weighted average of FPR
        precision = np.append(precision, macro_pre)
        recall = np.append(recall, macro_rec)
        f1 = np.append(f1, macro_f1)

        # Calculate weighted average of FPR
        weighted_f1 = f1_score(gt_labels, pred_labels, average="weighted")
        weighted_pre = precision_score(gt_labels, pred_labels, average="weighted")
        weighted_rec = recall_score(gt_labels, pred_labels, average="weighted")
        # Print the results for weighted average of FPR
        precision = np.append(precision, weighted_pre)
        recall = np.append(recall, weighted_rec)
        f1 = np.append(f1, weighted_f1)

        data = {"Precision": precision, "Recall": recall, "F1": f1}
        df = pd.DataFrame(data)

        # Display the DataFrame in the notebook
        print(df, file=log_f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pass command line arguments to the script"
    )
    parser.add_argument("best_dir", type=str, help="Path to the prompts file")
    parser.add_argument(
        "--dataset",
        type=str,
        default="fusrs_v2",
        help="Dataset to analyze, will affect ann path and num_categories",
    )
    parser.add_argument(
        "--log",
        type=str,
        default="./log.txt",
        help="To append to which log file",
    )

    args = parser.parse_args()

    if args.dataset == "fusrs_v2":
        ann = "./data/fusrs_v2/meta/test.txt"
    else:
        raise Exception("Dataset not supported")

    main(args.best_dir, ann, log_dir=args.log)
