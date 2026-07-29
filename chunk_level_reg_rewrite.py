from pathlib import Path
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import argparse


def sort_dict_by_keys(dictionary, key_list):
    """
    Sorts a dictionary by the keys in the provided list.

    :param dictionary: A dictionary (str -> Any)
    :param key_list: A list of keys (str) in the desired sort order
    :return: A list of values from the dictionary, sorted by the keys in key_list
    """
    if not set(key_list).issubset(dictionary.keys()):
        raise ValueError("All keys in key_list must be present in the dictionary")

    sorted_items = sorted(dictionary.items(), key=lambda item: key_list.index(item[0]))
    return [item[1] for item in sorted_items]


def construct_dataframe(fp):
    """
    Constructs a DataFrame from the output of a previous processing step (e.g., from chunk_level_detect.py). The input file is expected
    to be in JSON format.
    :param fp: The path to the input file (JSON).
    :return: tuple[dict,dict] with keys: first dict[hallucination_label_concat: A 2D array of hallucination labels (concatenated),
    external_similarity_concat: A 2D array of external similarity scores (concatenated), parameter_knowledge_concat: A 2D array of
    parameter knowledge scores (concatenated)].
    second dict[copy_heads_order: A list of sorted copy head indices, layer_order: A list of sorted layer indices, copy_heads: The
    original copy heads from the input data]
    """
    with Path(fp).open() as f:
        data: dict = json.load(f)
        copy_heads: list[tuple[int, int]] = data.pop("copy_heads")

    # this section makes sure all datapoints are at the exact same position. Because of loading we cant guarantee that the order of the
    # dictionary is always the same.
    copy_head_key_str = [str((i1, i2)) for i1, i2 in copy_heads]
    layer_key_str = list(
        data[list(data.keys())[0]]['scores'][list(data[list(data.keys())[0]]['scores'].keys())[0]]['parameter_knowledge_scores'].keys())

    # collecting import values, with the same order.
    prompt_attention_score = []
    parameter_knowledge_scores = []
    hallucination_label = []
    for k, v in data.items():
        for s in v['scores'].values():
            prompt_attention_score.append(sort_dict_by_keys(s["prompt_attention_score"], copy_head_key_str))
            parameter_knowledge_scores.append(sort_dict_by_keys(s['parameter_knowledge_scores'], layer_key_str))
            hallucination_label.append(s['hallucination_label'])

    prompt_attention_arr = np.array(prompt_attention_score).T
    parameter_knowledge_arr = np.array(parameter_knowledge_scores).T
    hallucination_label_arr = np.array(hallucination_label)
    response = {"statics": data}
    response.update(**{
        "hallucination_label_concat": hallucination_label_arr,
        "external_similarity_concat": prompt_attention_arr,
        "parameter_knowledge_concat": parameter_knowledge_arr,
    })
    info = {"copy_heads_order": copy_head_key_str, "layer_order": layer_key_str, "copy_heads": copy_heads}
    return response, info


def linear_regression(df):
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


def auc_pearson(y_true, y_score):
    """
    Calculates both the AUC and the Pearson correlation coefficient between y_true (actual labels) and y_score (predicted probabilities)

    :param y_true:
    :param y_score:
    :return: tuple[AUC, PEARSON]
    """
    auc = roc_auc_score(y_true, y_score)
    p = np.corrcoef(y_score, y_true, dtype=float)[-1][0]
    return auc, p


def min_max_normalization(x: np.ndarray):
    """
    min-max normalization.
    :param x:
    :return: normalized x
    """
    return (x - x.min()) / (x.max() - x.min())


def calculate_auc_pcc(dc: dict):
    """
    calculates the AUC and pearson coeff. for each layer on its own.
    :param dc:
    :return:
    """
    inv_labels = 1 - dc["hallucination_label_concat"]
    auc_ext_sim_list, pearson_ext_sim_list = np.array([auc_pearson(inv_labels, row) for row in dc["external_similarity_concat"][:-1]]).T
    auc_param_know_list, pearson_param_know_list = np.array(
        [auc_pearson(inv_labels, row) for row in dc["parameter_knowledge_concat"][:-1]]).T
    return auc_ext_sim_list, pearson_ext_sim_list, auc_param_know_list, pearson_param_know_list


def calculate_auc_pcc_32_32(dc: dict, copy_heads: list, auc_ext_arr: np.ndarray[float | int], auc_param_arr: np.ndarray[float | int],
                            top_n_heads: int = 3,
                            top_n_layers: int = 3, external_sim_scaling: float = 0.2, param_know_scaling=1):
    """
    calculates the complete AUC and pearson over all layers and heads.
    :param dc:
    :param copy_heads:
    :param auc_ext_arr:
    :param auc_param_arr:
    :param top_n_heads:
    :param top_n_layers:
    :param external_sim_scaling:
    :param param_know_scaling:
    :return: final AUC caluclations
    """
    results = {}
    # Sort by AUC and select the top N features (for example, top 5)
    top_n_auc_external_similarity_indx = auc_ext_arr.argsort(descending=True)[:top_n_heads]
    top_k_auc_parameter_knowledge_difference_indx = auc_param_arr.argsort(descending=True)[:top_n_layers]

    sorted_copy_heads = np.sort(np.array(copy_heads), axis=0)
    results.update({
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
    results.update({
        "head_max_min": (external_similarity_sum.max(), external_similarity_sum.min()),
        "layers_max_min": (parameter_knowledge_difference_sum.max(), parameter_knowledge_difference_sum.min()),
    })
    # Subtract the normalized columns
    difference_normalized = (
            param_know_scaling * parameter_knowledge_difference_sum_normalized - external_sim_scaling *
            external_similarity_sum_normalized)

    # Calculate AUC for the difference
    auc_difference_normalized = auc_pearson(dc['hallucination_label_concat'], difference_normalized)
    results.update({"Normalized Difference (AUX, PEARSON)": auc_difference_normalized})

    # this splits auc_difference_normalized into its corresponding chunks.
    split_lengths = [len(dc['statics'][k]['scores']) for k in dc['statics'].keys()]
    split_points = np.cumsum(split_lengths)[:-1]
    difference_normalized_split = np.split(difference_normalized, split_points)
    hallucination_label_split = np.split(dc['hallucination_label_concat'], split_points)
    difference_normalized_mean = np.array([np.mean(arr) for arr in difference_normalized_split])  # taking mean of correlating chunks
    hallucination_label = np.array(
        [np.max(arr) for arr in hallucination_label_split])  # # assuming sentence is hallucinated if at least one chunk is hallucinated

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
    """test arguments"""
    args = argparse.Namespace()
    args.dataset_path = "./test_output_chunk.json"
    args.output = "./test_output_chunk_results.json"
    args.top_n_heads = 22
    args.top_n_layers = 10
    args.external_sim_scaling = 0.2
    args.param_know_scaling = 1
    return args


def main(args: argparse.Namespace):
    dc, info = construct_dataframe(args.dataset_path)
    auc_ext_sim_arr, pearson_ext_sim_arr, auc_param_know_arr, pearson_param_know_arr = calculate_auc_pcc(dc)

    auc_difference_normalized, results = calculate_auc_pcc_32_32(dc=dc, copy_heads=info["copy_heads"],
                                                                 auc_ext_arr=auc_ext_sim_arr,
                                                                 auc_param_arr=auc_param_know_arr,
                                                                 top_n_heads=args.top_n_heads,
                                                                 top_n_layers=args.top_n_layers,
                                                                 external_sim_scaling=args.external_sim_scaling,
                                                                 param_know_scaling=args.param_know_scaling)

    result_dict = {"auc": auc_difference_normalized[0], "pcc": auc_difference_normalized[1], **results}
    with Path(args.output).open("w") as f:
        json.dump(result_dict, f, ensure_ascii=False)


if __name__ == "__main__":
    args = parse_arguments()
    # args = test_args()
    main(args)
