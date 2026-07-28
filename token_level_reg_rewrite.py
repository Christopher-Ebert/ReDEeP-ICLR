import pandas as pd
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import argparse
from pathlib import Path


def load_data(fp):
    with Path(fp).open() as f:
        data = json.load(f)
    return data


def construct_dataframe(fp) -> tuple[dict[str, np.ndarray], dict]:
    # Sample data for illustration
    with Path(fp).open() as f:
        data: dict = json.load(f)
        info = data.pop("info")
    response = {"statics": data}
    # print("hallucination_label value_counts:", df["hallucination_label"].value_counts(normalize=True))
    external_similarity_concat = np.concatenate([v["external_similarity"] for k, v in data.items()], axis=0).T
    parameter_knowledge_concat = np.concatenate([v["parameter_knowledge_difference"] for k, v in data.items()], axis=0).T
    hallucination_label_concat = np.concatenate([v["hallucination_label"] for k, v in data.items()], axis=0)
    response.update({
        "external_similarity_concat": external_similarity_concat,
        "parameter_knowledge_concat": parameter_knowledge_concat,
        "hallucination_label_concat": hallucination_label_concat,
    })
    return response, info


def linear_regression(df: pd.DataFrame):
    # Extract features and labels
    features = df.drop(columns=["identifier", "hallucination_label"])
    labels = df["hallucination_label"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Initialize and train the logistic regression model
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(accuracy)
    print(report)
    return accuracy, report  # TODO: maybe add f1, recall, precision


def calculate_auc_pcc(dc: dict) -> tuple[np.ndarray[float | int], ...]:
    inv_labels = 1 - dc["hallucination_label_concat"]
    auc_ext_sim_list = np.array([roc_auc_score(inv_labels, row) for row in dc["external_similarity_concat"][:-1]])
    pearson_ext_sim_list = np.corrcoef(dc["external_similarity_concat"][:-1], inv_labels)[:-1, -1]
    auc_param_know_list = np.array([roc_auc_score(inv_labels, row) for row in dc["parameter_knowledge_concat"][:-1]])
    pearson_param_know_list = np.corrcoef(dc["parameter_knowledge_concat"][:-1], inv_labels)[:-1, -1]
    return auc_ext_sim_list, pearson_ext_sim_list, auc_param_know_list, pearson_param_know_list


def min_max_normalization(x: np.ndarray):
    return (x - x.min()) / (x.max() - x.min())


def auc_pearson(y_true, y_score):
    auc = roc_auc_score(y_true, y_score)
    p = np.corrcoef(y_score, y_true, dtype=float)[-1][0]
    return auc, p


# copy_heads <-> [attn_layer, head]
def calculate_auc_pcc_32_32(dc: dict, copy_heads: list, auc_ext_arr: np.ndarray[float | int], auc_param_arr: np.ndarray[float | int],
                            top_n_heads: int = 3, top_n_layers: int = 3, external_sim_scaling: float = 0.2, param_know_scaling: int = 1):
    collect_info = {}
    # Sort by AUC and select the top N features (for example, top 5)
    top_n_auc_external_similarity_indx = auc_ext_arr.argsort(descending=True)[:top_n_heads]
    top_k_auc_parameter_knowledge_difference_indx = auc_param_arr.argsort(descending=True)[:top_n_layers]

    sorted_copy_heads = np.sort(np.array(copy_heads), axis=0)
    collect_info.update({
        "select_heads": sorted_copy_heads[top_n_auc_external_similarity_indx],
        "select_layers": top_k_auc_parameter_knowledge_difference_indx,
    })

    # Sum the top N features for each type
    external_similarity_sum = dc['external_similarity_concat'][top_n_auc_external_similarity_indx].T.sum(axis=1)
    parameter_knowledge_difference_sum = dc['parameter_knowledge_concat'][top_k_auc_parameter_knowledge_difference_indx].T.sum(axis=1)

    results = {
        "Top N External Similarity (AUX, PEARSON)":
            auc_pearson(1 - dc['hallucination_label_concat'], external_similarity_sum),
        "Top N Parameter Knowledge Difference (AUX, PEARSON)":
            auc_pearson(dc['hallucination_label_concat'], parameter_knowledge_difference_sum),
    }

    # Normalize the columns
    external_similarity_sum_normalized = min_max_normalization(external_similarity_sum)
    parameter_knowledge_difference_sum_normalized = min_max_normalization(parameter_knowledge_difference_sum)
    collect_info.update({
        "head_max_min": (external_similarity_sum.max(), external_similarity_sum.min()),
        "layers_max_min": (parameter_knowledge_difference_sum.max(), parameter_knowledge_difference_sum.min()),
    })
    # Subtract the normalized columns
    difference_normalized = (param_know_scaling * parameter_knowledge_difference_sum_normalized - external_sim_scaling *
                             external_similarity_sum_normalized)

    # Calculate AUC for the difference
    auc_difference_normalized = auc_pearson(dc['hallucination_label_concat'], difference_normalized)
    results.update({"Normalized Difference (AUX, PEARSON)": auc_difference_normalized})

    split_lengths = [len(dc['statics'][k]['external_similarity']) for k in dc['statics'].keys()]
    split_points = np.cumsum(split_lengths)[:-1]
    difference_normalized_split = np.split(difference_normalized, split_points)
    hallucination_label_split = np.split(dc['hallucination_label_concat'], split_points)
    difference_normalized_mean = np.array([np.mean(arr) for arr in difference_normalized_split])
    hallucination_label = np.array([np.max(arr) for arr in hallucination_label_split])

    difference_normalized_mean_norm = min_max_normalization(difference_normalized_mean)

    # Calculate AUC for the grouped means
    auc_difference_normalized_norm = auc_pearson(hallucination_label, difference_normalized_mean_norm)
    results.update({"Grouped means (AUX, PEARSON)": auc_difference_normalized_norm})
    return auc_difference_normalized_norm, results


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='ReDeEP token level Regression. Run this after token_level_detection.')
    parser.add_argument("-d", "--dataset_path", type=str, required=True, help=f"path to token_level_detection output")
    parser.add_argument("-o", "--output", type=str, default="./redeep_token_level_reg.json",
                        help="output path. Default: ./redeep_token_level_reg.json")
    parser.add_argument("-nh", "--top_n_heads", type=int, default=1, help="")
    parser.add_argument("-nl", "--top_n_layers", type=int, default=10)
    parser.add_argument("-a", "--external_sim_scaling", type=float, default=0.2)
    parser.add_argument("-m", "--param_know_scaling", type=int, default=1)
    return parser.parse_args()


def test_args():
    args = argparse.Namespace()
    args.dataset_path = "./redeep_llama27b.json"
    args.output = "./redeep_llama27b_regression_results.json"
    args.top_n = 22
    args.top_k = 10
    args.alpha = 0.2
    args.m = 1
    return args


def main(args: argparse.Namespace):
    # number = 32  # amount of samples. relates to layers of detect_model. No longer used, automatic methods used. Keeping for
    # explanation.
    dc, info = construct_dataframe(args.dataset_path)  # output of token_level_detect
    auc_ext_sim_arr, pearson_ext_sim_arr, auc_param_know_arr, pearson_param_know_arr = calculate_auc_pcc(dc)

    auc_difference_normalized, results = calculate_auc_pcc_32_32(dc=dc, copy_heads=info["copy_heads"],
                                                                 auc_ext_arr=auc_ext_sim_arr, auc_param_arr=auc_param_know_arr,
                                                                 top_n_heads=args.top_n_heads, top_n_layers=args.top_n_layers,
                                                                 external_sim_scaling=args.external_sim_scaling,
                                                                 param_know_scaling=args.param_know_scaling)

    result_dict = {"auc": auc_difference_normalized[0], "pcc": auc_difference_normalized[1], **results}
    with Path(args.output).open("w") as f:
        json.dump(result_dict, f, ensure_ascii=False)


if __name__ == "__main__":
    args = parse_arguments()
    # args = test_args()
    main(args)
