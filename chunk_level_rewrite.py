import argparse
import json
import warnings
from pathlib import Path
from typing import List, Tuple, Any, Iterable

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.nn import functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='ReDeEP chunk level.')
    parser.add_argument("-m", '--model_name', type=str, required=True, help='huggingface model identifyer')
    parser.add_argument("-d", "--dataset_path", type=str, required=True, help=f"path to dataset")
    parser.add_argument("-c", "--copy_heads_path", type=str, required=False, default=None,
                        help="topk heads to use as json_file.")
    parser.add_argument("--cache_dir", type=str, default="./cache_dir",
                        help="cache directory for saving superficial data")
    parser.add_argument("-t", "--token", type=str, help="huggingface token. can also be set using environmental.")
    parser.add_argument("-a", "--amount", type=int, default=-1, help="amount of datapoints to analyze")
    parser.add_argument("-k", "--knowledge_layers", required=False, nargs=2, default=[0, 32], help="knowledge layers")
    parser.add_argument("-o", "--output", type=str, default="./redeep_chunk_level_reg.json",
                        help="output path. Default: ./redeep_token_level_reg.json")
    parser.add_argument("-nh", "--top_n_heads", type=int, default=1, help="")
    parser.add_argument("-nl", "--top_n_layers", type=int, default=10)
    parser.add_argument("-es", "--external_sim_scaling", type=float, default=0.2)
    parser.add_argument("-pk", "--param_know_scaling", type=int, default=1)

    args = parser.parse_args()
    args.knowledge_layers = [int(i) for i in args.knowledge_layers]
    return args


class JsonEncoder(json.JSONEncoder):
    """
    json encoder allowing for serialization of pydantic and exception objects.
    """

    def default(self, o):
        if isinstance(o, torch.Tensor):
            return o.tolist()
        if isinstance(o, bool):
            return int(o)
        return super().default(o)


def load_data(fp, amount: int = -1, ):
    with Path(fp).open() as f:
        data: dict = json.load(f)

    # handling amount
    if amount == -1:
        return data
    avail_keys = list(data.keys())[:amount]
    data = {k: data[k] for k in avail_keys}
    return data


def load_copy_heads(fp) -> tuple[list, str]:
    data = load_data(fp)
    return data['copy_heads'], data["model"]


def load_model_and_tokenizer(model_name, cache_dir, hf_token) -> Tuple:
    """Load the model, tokenizer, and optional tokenizer for template."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        cache_dir=cache_dir,
        attn_implementation="eager",
        token=hf_token
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, token=hf_token)
    return model, tokenizer


def add_special_template(prompt: str, tokenizer: Any) -> str:
    """Add special chat template to the prompt."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return text


def calculate_dist(sep_vocabulary_dist: torch.Tensor, sep_attention_dist: torch.Tensor) -> float:
    """Calculate Jensen-Shannon divergence between two distributions."""
    softmax_mature_layer = F.softmax(sep_vocabulary_dist, dim=-1)
    softmax_anchor_layer = F.softmax(sep_attention_dist, dim=-1)

    M = 0.5 * (softmax_mature_layer + softmax_anchor_layer)

    log_softmax_mature_layer = F.log_softmax(sep_vocabulary_dist, dim=-1)
    log_softmax_anchor_layer = F.log_softmax(sep_attention_dist, dim=-1)

    kl1 = F.kl_div(log_softmax_mature_layer, M, reduction='none').mean(-1)
    kl2 = F.kl_div(log_softmax_anchor_layer, M, reduction='none').mean(-1)
    js_divs = 0.5 * (kl1 + kl2)

    return js_divs.cpu().item() * 10e5


# TODO: check this
def calculate_dist_2d(sep_vocabulary_dist: torch.Tensor, sep_attention_dist: torch.Tensor) -> float:
    """Calculate 2D Jensen-Shannon divergence between two distributions."""
    softmax_mature_layer = F.softmax(sep_vocabulary_dist, dim=-1)
    softmax_anchor_layer = F.softmax(sep_attention_dist, dim=-1)

    M = 0.5 * (softmax_mature_layer + softmax_anchor_layer)

    log_softmax_mature_layer = F.log_softmax(sep_vocabulary_dist, dim=-1)
    log_softmax_anchor_layer = F.log_softmax(sep_attention_dist, dim=-1)

    kl1 = F.kl_div(log_softmax_mature_layer, M, reduction='none').sum(dim=-1)
    kl2 = F.kl_div(log_softmax_anchor_layer, M, reduction='none').sum(dim=-1)
    js_divs = 0.5 * (kl1 + kl2)

    scores = js_divs.cpu().tolist()
    return sum(scores) if isinstance(scores, list) else scores


def calculate_ma_dist(sep_vocabulary_dist: torch.Tensor, sep_attention_dist: torch.Tensor) -> float:
    """Calculate Manhattan distance between two distributions."""
    sep_vocabulary_dist = F.softmax(sep_vocabulary_dist, dim=-1)

    dist_diff = sep_vocabulary_dist - sep_attention_dist
    abs_diff = torch.abs(dist_diff)
    manhattan_distance = torch.sum(abs_diff)

    return manhattan_distance.cpu().item()


def is_hallucination_token(token_id: int, hallucination_spans: List[List[int]]) -> bool:
    """Check if a token ID falls within any hallucination span."""
    for span in hallucination_spans:
        if span[0] <= token_id <= span[1]:
            return True
    return False


def is_hallucination_span(response_span: List[int], hallucination_spans: List[List[int]]) -> bool:
    """Check if any token in a response span falls within any hallucination span."""
    for token_id in range(response_span[0], response_span[1]):
        if is_hallucination_token(token_id, hallucination_spans):
            return True
    return False


def calculate_hallucination_spans(
        labels: Any,
        text: str,
        response_rag: str,
        tokenizer: Any,
) -> List[List[int]]:
    """Calculate hallucination spans in token IDs."""
    hallucination_span = []
    for item in labels:
        start_id = item['start']
        end_id = item['end']
        start_text = text + response_rag[:start_id]
        end_text = text + response_rag[:end_id]
        start_text_id = tokenizer(start_text, return_tensors="pt").input_ids
        end_text_id = tokenizer(end_text, return_tensors="pt").input_ids
        start_id = start_text_id.shape[-1]
        end_id = end_text_id.shape[-1]
        hallucination_span.append([start_id, end_id])
    return hallucination_span


def calculate_respond_spans(
        raw_response_spans: List[List[int]],
        text: str,
        response_rag: str,
        tokenizer: Any,
) -> List[List[int]]:
    """Calculate response spans in token IDs."""
    respond_spans = []
    for item in raw_response_spans:
        start_id = item[0]
        end_id = item[1]
        start_text = text + response_rag[:start_id]
        end_text = text + response_rag[:end_id]
        start_text_id = tokenizer(start_text, return_tensors="pt").input_ids
        end_text_id = tokenizer(end_text, return_tensors="pt").input_ids
        start_id = start_text_id.shape[-1]
        end_id = end_text_id.shape[-1]
        respond_spans.append([start_id, end_id])
    return respond_spans


def calculate_prompt_spans(
        raw_prompt_spans: List[List[int]],
        prompt: str,
        tokenizer: Any,
) -> List[List[int]]:
    """Calculate prompt spans in token IDs."""
    prompt_spans = []
    for item in raw_prompt_spans:
        start_id = item[0]
        end_id = item[1]
        start_text = prompt[:start_id]
        end_text = prompt[:end_id]
        added_start_text = add_special_template(start_text, tokenizer)
        added_end_text = add_special_template(end_text, tokenizer)
        start_text_id = tokenizer(added_start_text, return_tensors="pt").input_ids.shape[-1] - 4
        end_text_id = tokenizer(added_end_text, return_tensors="pt").input_ids.shape[-1] - 4
        prompt_spans.append([start_text_id, end_text_id])
    return prompt_spans


def calculate_sentence_similarity(response_text: str, prompt_text: str, bge_model: SentenceTransformer) -> float:
    """Calculate sentence similarity using BGE model."""
    part_embedding = bge_model.encode([response_text], normalize_embeddings=True)
    q_embeddings = bge_model.encode([prompt_text], normalize_embeddings=True)

    scores_named = np.matmul(q_embeddings, part_embedding.T).flatten()
    return float(scores_named[0])


def process_responses(
        dataset: dict[str, dict[str, Any]],
        model: Any,
        tokenizer: Any,
        copy_heads: Iterable[Iterable[int]],
        bge_model: SentenceTransformer,
        knowledge_layers: List[int],
) -> dict[str, dict[str, Any]]:
    dc = {}
    dc["copy_heads"] = copy_heads
    for dataset_key, dataset_value in tqdm(dataset.items(), desc="processing ReDeEP chunk level detection."):
        torch.cuda.empty_cache()
        response_rag = dataset_value['response']
        prompt = dataset_value['prompt']
        prompt_spans = dataset_value["prompt_spans"]
        original_prompt_spans = dataset_value['prompt_spans']
        original_response_spans = dataset_value['response_spans']
        labels: List | Tuple = dataset_value["labels"]

        text = add_special_template(prompt[:12000], tokenizer)
        input_text = text + response_rag

        input_ids = tokenizer([input_text], return_tensors="pt").input_ids
        # prefix_ids = tokenizer([text], return_tensors="pt").input_ids
        # continue_ids = input_ids[0, prefix_ids.shape[-1]:]  # not used in original code base as well

        hallucination_spans = []
        if labels is not None or len(labels) != 0:
            hallucination_spans = calculate_hallucination_spans(labels, text, response_rag, tokenizer)

        prompt_spans = calculate_prompt_spans(prompt_spans, prompt, tokenizer, )
        respond_spans = calculate_respond_spans(original_response_spans, text, response_rag, tokenizer)

        with torch.no_grad():
            logits_dict, outputs = model(
                input_ids=input_ids.to(model.device),
                output_attentions=True,
                output_hidden_states=True,
                knowledge_layers=list(range(knowledge_layers[0], knowledge_layers[1]))  # use custom compiled transformer library
            )

        logits_dict = {key: [value[0], value[1]] for key, value in logits_dict.items()}
        score_dict = {}
        for response_id, response_span in enumerate(respond_spans):
            layer_head_span = {}
            # assumes that attn_layer and head exist in model
            for attn_layer_id, head_id in copy_heads:
                scores = []  # p_span_score_dict. only saving mapping score, not p_span
                # Step 1, Eq.2 identify attended tokens
                for prompt_span in prompt_spans:
                    attention_score = outputs.attentions[attn_layer_id][0, head_id, :, :]
                    _score = torch.sum(attention_score[response_span[0]:response_span[1], prompt_span[0]:prompt_span[1]]).cpu().item()
                    scores.append(_score)
                p_id = scores.index(
                    max(scores))  # prompt_spans[scores.index(max(scores))] # Extrahieren Sie das p_span, das dem höchsten Wert entspricht.
                prompt_span_text = prompt[original_prompt_spans[p_id][0]:original_prompt_spans[p_id][1]]
                respond_span_text = response_rag[original_response_spans[response_id][0]:original_response_spans[response_id][1]]
                layer_head_span[str((attn_layer_id, head_id))] = calculate_sentence_similarity(prompt_span_text, respond_span_text,
                                                                                               bge_model)
            parameter_knowledge_scores = [
                calculate_dist_2d(value[0][0, response_span[0]:response_span[1], :], value[1][0, response_span[0]:response_span[1], :]) for
                value in logits_dict.values()]
            parameter_knowledge_dict = {f"layer_{i}": value for i, value in enumerate(parameter_knowledge_scores)}

            score_dict[response_id] = {
                "prompt_attention_score": layer_head_span,
                "response_span": response_span,
                "hallucination_label": 1 if is_hallucination_span(response_span, hallucination_spans) else 0,
                "parameter_knowledge_scores": parameter_knowledge_dict,
            }
        dc[dataset_key] = {"key": dataset_key, "scores": score_dict, **dataset_value}
    return dc


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


def construct_dataframe(processed_responses: dict[str, dict[str, Any]]):
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

    data = processed_responses
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

    # parameter_knowledge_difference_sum = ∑ P^l_t, l in F, external_similarity_sum = ∑E^l,h_t, l,h in A, $4.1
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
    # multivariate analysis approach that regresses decoupled External Context Score and Parametric Knowledge Score
    # H_t(r) = 1/|r| * sum(H_t(t), t in r), H_t(t) = α * sum(P^l_t, l in F) - β * sum(E^l,h_t, l,h in A), sum already computed, $4.1
    hallucination_score_difference_normalized = (param_know_scaling * parameter_knowledge_difference_sum_normalized - external_sim_scaling *
                                                 external_similarity_sum_normalized)

    # Calculate AUC for the difference
    auc_difference_normalized = auc_pearson(dc['hallucination_label_concat'], hallucination_score_difference_normalized)
    results.update({"Normalized Difference (AUX, PEARSON)": auc_difference_normalized})

    # this splits auc_difference_normalized into its corresponding chunks.
    split_lengths = [len(dc['statics'][k]['scores']) for k in dc['statics'].keys()]
    split_points = np.cumsum(split_lengths)[:-1]
    difference_normalized_split = np.split(hallucination_score_difference_normalized, split_points)
    hallucination_label_split = np.split(dc['hallucination_label_concat'], split_points)
    difference_normalized_mean = np.array([np.mean(arr) for arr in difference_normalized_split])  # taking mean of correlating chunks
    hallucination_label = np.array(
        [np.max(arr) for arr in hallucination_label_split])  # # assuming sentence is hallucinated if at least one chunk is hallucinated

    difference_normalized_mean_norm = min_max_normalization(difference_normalized_mean)

    # Calculate AUC for the grouped means
    auc_difference_normalized_norm = auc_pearson(hallucination_label, difference_normalized_mean_norm)
    results.update({"Grouped means (AUX, PEARSON)": auc_difference_normalized_norm})
    return auc_difference_normalized_norm, results


def custom_print(s: str, symbol: str = '#', amount: int = 10):
    print(symbol * amount + ' ' + s + ' ' + symbol * amount)


def step1(args: argparse.Namespace):
    """Main function to orchestrate the processing pipeline."""
    # setup
    copy_heads, copy_heads_model = load_copy_heads(args.copy_heads_path)
    if args.model_name != copy_heads_model:
        warnings.warn(
            f"provided copy_heads file was created with different model as currently provided. Please check that this is expected. "
            f"model_name={args.model_name} copy_heads_model={copy_heads_model}")

    dataset = load_data(args.dataset_path, args.amount)
    model, tokenizer = load_model_and_tokenizer(args.model_name, args.cache_dir, args.token)

    bge_model = SentenceTransformer("BAAI/bge-base-en-v1.5", cache_folder=args.cache_dir, token=args.token).to("cuda")
    custom_print('step 1. processing responses.')
    processed_responses = process_responses(
        dataset,
        model,
        tokenizer,
        copy_heads,
        bge_model,
        args.knowledge_layers
    )
    return processed_responses


def step2(args: argparse.Namespace, processed_responses):
    custom_print('step 2. analysing responses.')
    dc, info = construct_dataframe(processed_responses)
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
    processed_responses = step1(args)
    step2(args, processed_responses)
    custom_print('done')
