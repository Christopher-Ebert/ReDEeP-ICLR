import pandas as pd
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler
import pdb
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from tqdm import tqdm
import argparse
from pathlib import Path
from typing import Any


def load_data(fp):
    with Path(fp).open() as f:
        data = json.load(f)
    return data


def construct_dataframe(fp) -> tuple[dict, dict]:
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


def calculate_auc_pcc(dc: dict):
    inv_labels = 1 - dc["hallucination_label_concat"]
    auc_ext_sim_list = [roc_auc_score(inv_labels, row) for row in dc["external_similarity_concat"][:-1]]
    pearson_ext_sim_list = np.corrcoef(dc["external_similarity_concat"][:-1], inv_labels)[:-1, -1]
    auc_param_know_list = [roc_auc_score(inv_labels, row) for row in dc["parameter_knowledge_concat"][:-1]]
    pearson_param_know_list = np.corrcoef(dc["parameter_knowledge_concat"][:-1], inv_labels)[:-1, -1]
    return auc_ext_sim_list, pearson_ext_sim_list, auc_param_know_list, pearson_param_know_list


# copy_heads <-> [attn_layer, head]
def calculate_auc_pcc_32_32(dc: dict, copy_heads: list, top_n: int, top_k: int, alpha: float,
                            auc_ext_sim_list: list, auc_param_know_list: list, m: int = 1):
    collect_info = {}
    # Sort by AUC and select the top N features (for example, top 5)
    top_n_auc_external_similarity = sorted(auc_ext_sim_list, reverse=True)[:top_n]
    top_k_auc_parameter_knowledge_difference = sorted(auc_param_know_list, reverse=True)[:top_k]

    # top layers
    top_ext_index = [auc_ext_sim_list.index(i) for i in top_n_auc_external_similarity]
    top_param_index = [auc_param_know_list.index(i) for i in top_k_auc_parameter_knowledge_difference]

    sorted_copy_heads = sorted(copy_heads, key=lambda x: (x[0], x[1]))
    collect_info.update({
        "select_heads": [sorted_copy_heads[idx] for idx in top_ext_index],
        "select_layers": top_param_index
    })

    # Sum the top N features for each type
    external_similarity_sum = np.array([dc['external_similarity_concat'][col] for col in top_ext_index]).T.sum(axis=1)
    parameter_knowledge_difference_sum = np.array(
        [dc['parameter_knowledge_concat'][col] for col in top_param_index]).T.sum(axis=1)

    results = {
        "Top N AUC External Similarity": roc_auc_score(1 - dc['hallucination_label_concat'], external_similarity_sum),
        "Top N AUC Parameter Knowledge Difference": roc_auc_score(dc['hallucination_label_concat'],
                                                                  parameter_knowledge_difference_sum),
        "Top N Pearson Correlation External Similarity": pearsonr(external_similarity_sum,
                                                                  1 - dc['hallucination_label_concat']),
        "Top N Pearson Correlation Parameter Knowledge Difference": pearsonr(parameter_knowledge_difference_sum,
                                                                             dc['hallucination_label_concat'])
    }

    scaler1 = MinMaxScaler()
    scaler2 = MinMaxScaler()
    # Normalize the columns
    external_similarity_sum_normalized = scaler1.fit_transform(external_similarity_sum.reshape(-1,1))
    parameter_knowledge_difference_sum_normalized = scaler2.fit_transform(parameter_knowledge_difference_sum.reshape(-1,1))
    collect_info.update({
        "head_max_min": [scaler1.data_max_[0], scaler1.data_min_[0]],
        "layers_max_min": [scaler2.data_max_[0], scaler2.data_min_[0]]
    })
    # Subtract the normalized columns
    difference_normalized = m * parameter_knowledge_difference_sum_normalized - alpha * external_similarity_sum_normalized

    # Calculate AUC for the difference
    auc_difference_normalized = roc_auc_score(dc['hallucination_label_concat'], difference_normalized)
    person_difference_normalized, _ = pearsonr(dc['hallucination_label_concat'], difference_normalized.reshape(-1))
    results.update({
        "Normalized Difference AUC": auc_difference_normalized,
        "Normalized Difference Pearson Correlation": person_difference_normalized
    })


    for k in dc['statics'].keys():
        amount_values = len(dc['statics'][k]["'external_similarity'"])
        hallucination_label =


    # Group by 'identifier' and calculate the sum of 'difference_normalized' and max of 'hallucination_label'
    dc['response_group'] = dc['identifier'].str.extract(r'(response_\d+)')
    # Group by 'response_group' and calculate the sum of 'difference_normalized' and max of 'hallucination_label'
    grouped_df = dc.groupby('response_group').agg(
        difference_normalized_mean=('difference_normalized', 'mean'),
        hallucination_label=('hallucination_label', 'max')
    ).reset_index()

    min_val = grouped_df['difference_normalized_mean'].min()
    max_val = grouped_df['difference_normalized_mean'].max()
    collect_info.update({'final_max_min': [max_val, min_val]})
    # 进行归一化
    grouped_df['difference_normalized_mean_norm'] = (grouped_df['difference_normalized_mean'] - min_val) / (
            max_val - min_val)

    # Calculate AUC for the grouped means
    auc_difference_normalized = roc_auc_score(grouped_df['hallucination_label'],
                                              grouped_df['difference_normalized_mean_norm'])
    person_difference_normalized, _ = pearsonr(grouped_df['hallucination_label'],
                                               grouped_df['difference_normalized_mean_norm'])

    results.update({"Grouped means AUC": auc_difference_normalized})
    results.update({"Grouped means Pearson Correlation": person_difference_normalized})
    return auc_difference_normalized, person_difference_normalized


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='ReDeEP token level detection.')
    parser.add_argument("-m", '--model_name', type=str, required=True, help='huggingface model identifyer')
    parser.add_argument("-d", "--dataset_path", type=str, required=True, help=f"path to dataset")
    parser.add_argument("-o", "--output", type=str, default="./redeep_token_level_regression.json",
                        help="output path. Default: ./redeep_token_level_detection.json")  # TODO: do me.
    parser.add_argument("--cache_dir", type=str, default="./cache_dir",
                        help="cache directory for saving superficial data")
    parser.add_argument("--top_n", type=int, default=1, help="")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--m", type=int, default=1)
    return parser.parse_args()


def main(args: argparse.Namespace):
    # number = 32  # amount of samples. relates to layers of detect_model. No longer used, automatic methods used. Keeping for explanation.
    dc, info = construct_dataframe(args.dataset_path)  # output of token_level_detect
    auc_ext_sim_list, pearson_ext_sim_list, auc_param_know_list, pearson_param_know_list = calculate_auc_pcc(dc)

    auc_difference_normalized, person_difference_normalized = calculate_auc_pcc_32_32(dc, info["copy_heads"],
                                                                                      args.top_n, args.top_k,
                                                                                      args.alpha,
                                                                                      auc_ext_sim_list,
                                                                                      auc_param_know_list,
                                                                                      args.m)

    result_dict = {"auc": auc_difference_normalized, "pcc": person_difference_normalized}
    print(result_dict)
    with open(save_path, 'w') as f:
        json.dump(result_dict, f, ensure_ascii=False)


def test_args():
    args = argparse.Namespace()
    args.model_name = "meta-llama/Llama-2-7b-chat-hf"
    args.dataset_path = "./test_output.json"
    args.output = "./test_regression_results.json"
    args.cache_dir = "./.cache_dir"
    args.top_n = 1
    args.top_k = 10
    args.alpha = 0.2
    args.m = 1
    return args


if __name__ == "__main__":
    # args = parse_arguments()
    args = test_args()
    main(args)
