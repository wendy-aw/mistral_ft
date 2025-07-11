from sklearn.metrics import (
    hamming_loss,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    jaccard_score,
)
from sklearn.preprocessing import MultiLabelBinarizer
import json
import argparse
import os
import csv

# fmt: off
ALL_POSS_IDS = [
    "A01","A47","A61","A63","B01","B23","B25","B29","B32","B33","B41","B60",
    "B62","B64","B65","C01","C07","C08","C09","C12","C23","E21","F01","F02",
    "F04","F05","F16","F21","F24","G01","G02","G03","G05","G06","G07","G08",
    "G09","G10","G11","G16","H01","H02","H03","H04","H05","H10","Y02","Y10"
    ]
# fmt: on

OUTPUT_FILE_PATH = "metrics.csv"


def compute_hallucination_rate(
    pred_ids_list: list[list[str]], all_poss_ids: list[str]
) -> float:
    all_poss_set = set(all_poss_ids)
    total_preds = 0
    total_hallucinated = 0

    for preds in pred_ids_list:
        preds_set = set(preds)
        hallucinated = preds_set - all_poss_set  # Labels not in valid set
        total_hallucinated += len(hallucinated)
        total_preds += len(preds)

    if total_preds == 0:
        return 0.0  # Avoid division by zero

    return total_hallucinated / total_preds


def compute_metrics(
    true_ids_list: list[list[str]],
    pred_ids_list: list[list[str]],
    all_poss_ids: list[str],
) -> dict[str, float]:
    # Binarize labels for sklearn metrics
    mlb = MultiLabelBinarizer(classes=all_poss_ids)
    mlb.fit([all_poss_ids])
    true_bin = mlb.transform(true_ids_list)
    pred_bin = mlb.transform(pred_ids_list)

    # Hamming loss
    ham_loss = hamming_loss(true_bin, pred_bin)

    # Subset accuracy
    subset_acc = accuracy_score(true_bin, pred_bin)

    # Micro precision/recall/F1
    micro_prec = precision_score(true_bin, pred_bin, average="micro", zero_division=0)
    micro_rec = recall_score(true_bin, pred_bin, average="micro", zero_division=0)
    micro_f1 = f1_score(true_bin, pred_bin, average="micro", zero_division=0)

    # Macro precision/recall/F1
    if len(true_ids_list) == 1:
        macro_prec = None
        macro_rec = None
        macro_f1 = None
    else:
        macro_prec = precision_score(
            true_bin, pred_bin, average="macro", zero_division=0
        )
        macro_rec = recall_score(true_bin, pred_bin, average="macro", zero_division=0)
        macro_f1 = f1_score(true_bin, pred_bin, average="macro", zero_division=0)

    # Jaccard Index
    jaccard_index = jaccard_score(true_bin, pred_bin, average="samples")

    # Hallucination rate
    hallucination_rate = compute_hallucination_rate(pred_ids_list, all_poss_ids)

    # Num empty predictions
    num_empty = sum(1 for pred in pred_ids_list if len(pred) == 0)

    # Per-class F1-score
    per_class_f1 = f1_score(true_bin, pred_bin, average=None, zero_division=0)
    class_f1_dict = {
        cls: f"{round(f1, 3)}" for cls, f1 in zip(all_poss_ids, per_class_f1)
    }

    # Identify weak classes (F1 < 0.5)
    weak_classes = [cls for cls, f1 in class_f1_dict.items() if float(f1) < 0.5]

    return {
        "hamming_loss": f"{round(ham_loss, 3)}",
        "subset_acc": f"{round(subset_acc, 3)}",
        "micro_prec": f"{round(micro_prec, 3)}",
        "micro_rec": f"{round(micro_rec, 3)}",
        "micro_f1": f"{round(micro_f1, 3)}",
        "macro_prec": f"{round(macro_prec, 3)}" if macro_prec is not None else None,
        "macro_rec": f"{round(macro_rec, 3)}" if macro_rec is not None else None,
        "macro_f1": f"{round(macro_f1, 3)}" if macro_f1 is not None else None,
        "jaccard_index": f"{round(jaccard_index, 3)}",
        "hallucination_rate": f"{round(hallucination_rate, 3)}",
        "num_empty": num_empty,
        "weak_classes": json.dumps(weak_classes),
        "class_f1": json.dumps(class_f1_dict),
    }


def extract_pred_ids(
    true_ids_list: list[list[str]], results_path: str
) -> list[list[str]]:
    def extract_id_list(data: dict) -> list[str]:
        if "response" in data:
            return data["response"]["body"]["choices"][0]["message"]["content"].strip()
        elif "pred_class_ids" in data:
            return data["pred_class_ids"]
        else:
            raise ValueError(
                "Invalid results file format. Expected 'response' or 'pred_class_ids' key"
            )

    pred_ids_list = [[] for _ in range(len(true_ids_list))]
    with open(results_path, "r") as f:
        for line in f:
            data = json.loads(line)
            custom_id = int(data["custom_id"])
            id_list = extract_id_list(data)
            # Replace backticks with empty string
            id_list = id_list.replace("```json\n", "")
            id_list = id_list.replace("\n```", "")
            try:
                id_list = json.loads(id_list)
            except json.JSONDecodeError:
                # If ends with comma, replace with close bracket and retry
                if id_list.endswith(","):
                    id_list = id_list[:-1] + "]"
                try:
                    id_list = json.loads(id_list)
                except json.JSONDecodeError:
                    print("Invalid JSON:", id_list)
                    id_list = []
            try:
                pred_ids_list[custom_id] = (
                    id_list  # Because custom_id in results file is not in order
                )
            except IndexError:
                print("IndexError: custom_id", custom_id, "out of range")
    return pred_ids_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CPC labels")
    parser.add_argument(
        "--results", "-r", type=str, help="Path to JSONL inference results"
    )
    parser.add_argument(
        "--model", "-m", type=str, default="ministral-3b-2410", help="Model name"
    )
    parser.add_argument(
        "--test_data",
        "-t",
        type=str,
        default="data/df_test_final.jsonl",
        help="Path to JSONL test data with cpc_class_ids field",
    )
    parser.add_argument(
        "--ft_notes", "-f", type=str, default="", help="Notes on fine-tuning"
    )
    parser.add_argument(
        "--data_notes", "-d", type=str, default="", help="Notes on data"
    )
    args = parser.parse_args()

    # Validate args
    if not args.results:
        raise ValueError("Path to inference results file is not defined")
    if not os.path.exists(args.test_data):
        raise ValueError(f"Test data file {args.test_data} does not exist")
    # Check if results file has custom_id and pred_class_ids or response fields
    with open(args.results, "r") as f:
        first_line = f.readline()
        if (
            "custom_id" not in first_line
            and "pred_class_ids" not in first_line
            and "response" not in first_line
        ):
            raise ValueError(
                "Results file must have custom_id and pred_class_ids or response fields"
            )
    # Check if results file has the same number of entries as test data
    with open(args.results, "r") as f:
        num_lines = sum(1 for line in f)
    with open(args.test_data, "r") as f:
        if num_lines != sum(1 for line in f):
            raise ValueError(
                "Results file must have the same number of entries as test data"
            )

    # Get true ids list from data/df_test_final.jsonl
    true_ids_list = []
    with open(args.test_data, "r") as f:
        for line in f:
            true_ids_list.append(json.loads(line)["cpc_class_ids"])

    # Get pred ids list from args.results
    pred_ids_list = extract_pred_ids(true_ids_list, args.results)

    # Compute metrics
    metrics = compute_metrics(true_ids_list, pred_ids_list, ALL_POSS_IDS)
    print(json.dumps(metrics, indent=4))

    # Write metrics to metrics.csv
    fieldnames = [
        "model",
        "results_file",
        "ft_notes",
        "data_notes",
        "hamming_loss",
        "subset_acc",
        "micro_prec",
        "micro_rec",
        "micro_f1",
        "macro_prec",
        "macro_rec",
        "macro_f1",
        "jaccard_index",
        "hallucination_rate",
        "num_empty",
        "weak_classes",
        "class_f1",
    ]

    row = {
        "model": args.model,
        "results_file": args.results,
        "ft_notes": args.ft_notes,
        "data_notes": args.data_notes,
        "hamming_loss": metrics["hamming_loss"],
        "subset_acc": metrics["subset_acc"],
        "micro_prec": metrics["micro_prec"],
        "micro_rec": metrics["micro_rec"],
        "micro_f1": metrics["micro_f1"],
        "macro_prec": metrics["macro_prec"],
        "macro_rec": metrics["macro_rec"],
        "macro_f1": metrics["macro_f1"],
        "jaccard_index": metrics["jaccard_index"],
        "hallucination_rate": metrics["hallucination_rate"],
        "num_empty": metrics["num_empty"],
        "weak_classes": metrics["weak_classes"],
        "class_f1": metrics["class_f1"],
    }

    # Write header only if file is empty
    write_header = (
        not os.path.exists(OUTPUT_FILE_PATH) or os.path.getsize(OUTPUT_FILE_PATH) == 0
    )

    with open(OUTPUT_FILE_PATH, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
