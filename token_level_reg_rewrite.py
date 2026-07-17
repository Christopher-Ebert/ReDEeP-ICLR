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


def construct_dataframe(fp) -> tuple[pd.DataFrame, dict]:
    # Sample data for illustration
    with Path(fp).open() as f:
        response: dict = json.load(f)
    info = response.pop('info')
    df = pd.DataFrame(response)
    # print("hallucination_label value_counts:", df["hallucination_label"].value_counts(normalize=True))
    return df, info


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


def calculate_auc_pcc(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    # Calculate AUC and Pearson correlation for each of the 64 values
    dc: dict[str, dict[str, Any]] = {}
    for k, v in df.items():
        # External similarity metrics
        hallu_label = np.array(v['hallucination_label'])
        ext_sim = np.array(v['external_similarity']).T  # TODO: check transposing here is correct.
        param_know_diff = np.array(v['parameter_knowledge_difference']).T

        auc_ext_list = [roc_auc_score(1 - hallu_label, ext_sim[i]) for i in
                        range(ext_sim.shape[0] - 1)]  # not sure why y_true has to be the boolean-inverse.
        pearson_ext_list = [pearsonr(ext_sim[i], 1 - hallu_label).correlation for i in
                            range(ext_sim.shape[0] - 1)]  # TODO: check correlation is the correct value.

        auc_param = [roc_auc_score(hallu_label, param_know_diff[i]) for i in range(param_know_diff.shape[0] - 1)]
        pearson_param = [pearsonr(param_know_diff[i], hallu_label).correlation for i in range(param_know_diff.shape[0] - 1)]
        dc[k] = {"auc_external_similarity": auc_ext_list,
                 "pearson_external_similarity": pearson_ext_list,
                 "auc_parameter_knowledge_difference": auc_param,
                 "pearson_parameter_knowledge_difference": pearson_param
                 }
        # Parameter knowledge difference metrics
        # not sure what this debug-flag helps with. Keeping here for information purpose. This implementation does no longer work with current df-layout.
        # if v[f'parameter_knowledge_difference'].nunique() == 1:
        #     print(k)
    return dc


# copy_heads <-> [attn_layer, head]
def calculate_auc_pcc_32_32(df: pd.DataFrame, copy_heads: list[tuple[int, int]], top_n: int, top_k: int, alpha: float,
                            auc_pcc_dict: dict[str, dict[str, Any]], m: int = 1):
    collect_info = {}
    # Sort by AUC and select the top N features (for example, top 5)
    top_n_auc_external_similarity = sorted(auc_pcc_dict["auc_external_similarity"], reverse=True)[:top_n]
    top_k_auc_parameter_knowledge_difference = sorted(auc_pcc_dict["auc_parameter_knowledge_difference"], reverse=True)[
        :top_k]

    sorted_copy_heads = sorted(copy_heads, key=lambda x: (x[0], x[1]))
    collect_info.update(
        {"select_heads": [sorted_copy_heads[eval(name.split('_')[-1])] for _, name in top_n_auc_external_similarity]})

    if args.model_name == "llama2-13b":
        base_layer = 7
    else:
        base_layer = 0
    collect_info.update({"select_layers": [eval(name.split('_')[-1]) + base_layer for _, name in
                                           top_k_auc_parameter_knowledge_difference]})

    # Sum the top N features for each type
    df['external_similarity_sum'] = df[[col for _, col in top_n_auc_external_similarity]].sum(axis=1)
    df['parameter_knowledge_difference_sum'] = df[[col for _, col in top_k_auc_parameter_knowledge_difference]].sum(
        axis=1)

    # Calculate AUC for the summed top N features
    final_auc_external_similarity = roc_auc_score(1 - df['hallucination_label'], df['external_similarity_sum'])
    final_auc_parameter_knowledge_difference = roc_auc_score(df['hallucination_label'],
                                                             df['parameter_knowledge_difference_sum'])

    # Calculate Pearson correlation for the summed top N features
    final_pearson_external_similarity, _ = pearsonr(df['external_similarity_sum'], 1 - df['hallucination_label'])
    final_pearson_parameter_knowledge_difference, _ = pearsonr(df['parameter_knowledge_difference_sum'],
                                                               df['hallucination_label'])

    results = {
        "Top N AUC External Similarity": final_auc_external_similarity,
        "Top N AUC Parameter Knowledge Difference": final_auc_parameter_knowledge_difference,
        "Top N Pearson Correlation External Similarity": final_pearson_external_similarity,
        "Top N Pearson Correlation Parameter Knowledge Difference": final_pearson_parameter_knowledge_difference
    }

    scaler = MinMaxScaler()
    # Normalize the columns
    df['external_similarity_sum_normalized'] = scaler.fit_transform(df[['external_similarity_sum']])
    external_similarity_sum_max_value = scaler.data_max_[0]
    external_similarity_sum_min_value = scaler.data_min_[0]
    collect_info.update({
        "head_max_min": [external_similarity_sum_max_value, external_similarity_sum_min_value],
    })
    df['parameter_knowledge_difference_sum_normalized'] = scaler.fit_transform(
        df[['parameter_knowledge_difference_sum']])
    parameter_knowledge_sum_max_value = scaler.data_max_[0]
    parameter_knowledge_sum_min_value = scaler.data_min_[0]
    collect_info.update({
        "layers_max_min": [parameter_knowledge_sum_max_value, parameter_knowledge_sum_min_value]
    })
    # Subtract the normalized columns
    df['difference_normalized'] = m * df['parameter_knowledge_difference_sum_normalized'] - alpha * df[
        'external_similarity_sum_normalized']

    # Calculate AUC for the difference
    auc_difference_normalized = roc_auc_score(df['hallucination_label'], df['difference_normalized'])
    person_difference_normalized, _ = pearsonr(df['hallucination_label'], df['difference_normalized'])
    results.update({"Normalized Difference AUC": auc_difference_normalized})
    results.update({"Normalized Difference Pearson Correlation": person_difference_normalized})

    # Group by 'identifier' and calculate the sum of 'difference_normalized' and max of 'hallucination_label'
    df['response_group'] = df['identifier'].str.extract(r'(response_\d+)')

    # Group by 'response_group' and calculate the sum of 'difference_normalized' and max of 'hallucination_label'
    grouped_df = df.groupby('response_group').agg(
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
    df, info = construct_dataframe(args.dataset_path)  # output of token_level_detect
    auc_pcc_dict: dict[str, dict[str, Any]] = calculate_auc_pcc(df)

    auc_difference_normalized, person_difference_normalized = calculate_auc_pcc_32_32(df, info['copy_heads'],
                                                                                      args.top_n, args.top_k,
                                                                                      args.alpha,
                                                                                      auc_pcc_dict,
                                                                                      args.m)

    result_dict = {"auc": auc_difference_normalized, "pcc": person_difference_normalized}
    # print(result_dict)
    # with open(save_path, 'w') as f:
    #     json.dump(result_dict, f, ensure_ascii=False)


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
