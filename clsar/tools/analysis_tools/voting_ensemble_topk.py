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
import re
from collections import Counter


def vot_ensemble(
    pred_labels: np.ndarray,
    gt_labels: np.ndarray,
    identifier: str,
    log_dir: str = "./log.txt",
    save: bool = True,
):
    """calculate the voting ensemble results and log them to a file

    Args:
        pred_labels (np.ndarray): predicted labels
        gt_labels (np.ndarray): ground truth labels
        identifier (str): an identifier for the current experiment
        log_dir (str, optional): directory to save the exp logs. Defaults to "./log.txt".
    """

    # Calculate the per-category precision, recall, and F1-score

    precision, recall, f1, _ = precision_recall_fscore_support(
        gt_labels, pred_labels, average=None
    )

    if save:
        with open(log_dir, "a") as log_f:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n************** {current_time} **************", file=log_f)
            print(f"Ensembled prediction for {identifier}", file=log_f)
            # Calculate unweighted average (Macro) of FPR
            # Calculate macro F1-score
            print("groundtruth: ", gt_labels)
            print("prediction: ", pred_labels)
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
    else:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n************** {current_time} **************")
        print(f"Ensembled prediction for {identifier}")
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
        print(df)


def main_vot(
    ckpts_dir: str,
    dataset: str = "fusrs_v2",
    log: str = "./log.txt",
    voting: int = 5,
    save: bool = True,
):
    if dataset == "fusrs_v2":
        ann = "./data/fusrs_v2/meta/test.txt"
    else:
        raise Exception("Dataset not supported")

        # Load ground truth labels from label.txt
    gt_labels = []
    with open(ann, "r") as f:
        for line in f:
            parts = line.strip().split()
            gt_labels.append(int(parts[-1]))

    f1_score_dict = {}
    for ckpt in os.listdir(ckpts_dir):
        if ckpt.endswith(".json") and re.match(r"top\d+_f1_score.*\.json", ckpt):
            ckpt = os.path.join(ckpts_dir, ckpt)
            with open(ckpt, "r") as f:
                pred_data = json.load(f)
                pred_labels = pred_data["pred_label"]
            macro_f1 = f1_score(gt_labels, pred_labels, average="macro")
            f1_score_dict[ckpt] = macro_f1

    n = voting  # You can set your desired value for n
    top_n_models = sorted(f1_score_dict, key=f1_score_dict.get, reverse=True)[:n]

    # Load the predictions from the top n models
    top_n_predictions = []
    for model in top_n_models:
        with open(model, "r") as f:
            pred_data = json.load(f)
            pred_labels = pred_data["pred_label"]
        top_n_predictions.append(pred_labels)

    # Perform voting ensemble on the top n model predictions
    ensemble_predictions = []
    for preds in zip(*top_n_predictions):
        ensemble_predictions.append(Counter(preds).most_common(1)[0][0])

    # Calculate the F1 score for the ensemble predictions
    vot_ensemble(ensemble_predictions, gt_labels, ckpts_dir, log, save)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pass command line arguments to the script"
    )
    parser.add_argument("ckpts_dir", type=str, help="Path to the checkpoints dir")
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
    parser.add_argument(
        "--voting",
        type=int,
        default="5",
        help="To allow top n models for voting ensemble",
    )

    args = parser.parse_args()

    main_vot(
        ckpts_dir=args.ckpts_dir, dataset=args.dataset, log=args.log, voting=args.voting
    )
