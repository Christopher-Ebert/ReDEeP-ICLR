import warnings
from pathlib import Path
from typing import Any, Iterable, Dict, List, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from torch.nn import functional as F
import argparse
from tqdm import tqdm
import gc

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


def calculate_dist(sep_vocabulary_dist, sep_attention_dist):
    softmax_mature_layer = F.softmax(sep_vocabulary_dist, dim=-1)
    softmax_anchor_layer = F.softmax(sep_attention_dist, dim=-1)

    M = 0.5 * (softmax_mature_layer + softmax_anchor_layer)

    # 4. Calculate log-softmax for the KL divergence
    log_softmax_mature_layer = F.log_softmax(sep_vocabulary_dist, dim=-1)
    log_softmax_anchor_layer = F.log_softmax(sep_attention_dist, dim=-1)

    # 5. Calculate the KL divergences and then the JS divergences
    kl1 = F.kl_div(log_softmax_mature_layer, M, reduction='none').mean(-1)
    kl2 = F.kl_div(log_softmax_anchor_layer, M, reduction='none').mean(-1)
    # # Fix bug: https://github.com/Jeryi-Sun/ReDEeP-ICLR/issues/2 but for stable calculation, we maintain the original implementation of JSD.
    # kl1 = F.kl_div(M.log(), softmax_mature.unsqueeze(0),  reduction='none').mean(-1)
    # kl2 = F.kl_div(M.log(), softmax_anchor,  reduction='none').mean(-1)
    js_divs = 0.5 * (kl1 + kl2)

    return js_divs.cpu().item() * 10e5


def calculate_ma_dist(sep_vocabulary_dist, sep_attention_dist) -> float:
    sep_vocabulary_dist = F.softmax(sep_vocabulary_dist, dim=-1)

    dist_diff = sep_vocabulary_dist - sep_attention_dist
    # 取绝对值
    abs_diff = torch.abs(dist_diff)

    # 计算 Manhattan 距离
    manhattan_distance = torch.sum(abs_diff)

    return manhattan_distance.cpu().item()


def is_hallucination_token(token_id, hallucination_spans) -> bool:
    for span in hallucination_spans:
        if token_id >= span[0] and token_id <= span[1]:
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


def add_special_template(prompt: str, tokenizer: Any) -> str:
    """Add special chat template to the prompt."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return text


def process_responses(
        dataset: dict[str, dict[str, Any]],
        model: Any,
        tokenizer: Any,
        copy_heads: Iterable[Iterable[int]],
        knowledge_layers: List[int]
) -> dict[str, dict[str, Any]]:
    dc = {}
    for dataset_key, dataset_value in tqdm(dataset.items(), desc="processing ReDeEP token level detection."):
        response_rag = dataset_value['response']
        prompt = dataset_value['prompt']
        labels: List | Tuple = dataset_value["labels"]

        text = add_special_template(prompt[:12000], tokenizer)
        input_text = text + response_rag

        input_ids = tokenizer([input_text], return_tensors="pt").input_ids
        prefix_ids = tokenizer([text], return_tensors="pt").input_ids
        continue_ids = input_ids[0, prefix_ids.shape[
                                        -1]:]  # todo 这边要改成幻觉 token 的起止位置 -> This needs to be changed to the start and end positions of the hallucination tokens.

        hallucination_spans = []
        if labels is not None or len(labels) != 0:
            hallucination_spans = calculate_hallucination_spans(labels, text, response_rag, tokenizer)

        with torch.no_grad():
            logits_dict, outputs = model(
                input_ids=input_ids.to(model.device),
                output_attentions=True,
                output_hidden_states=True,
                knowledge_layers=list(range(knowledge_layers[0], knowledge_layers[1]))
            )

        logits_dict = {key: [value[0].to(model.device), value[1].to(model.device)] for key, value in
                       logits_dict.items()}
        # skip tokens without hallucination
        # outputs.hidden_states = tuple ([batch, seq_len, vocab_size], ..., )
        last_hidden_states = outputs.hidden_states[-1][0, :, :]  # [prefix_len, hidden_size]
        outputs.hidden_states = None  # memory optimization

        # todo 修改成 筛选 teacher focusing 的 token 和 model generate token 是否在 top_10内 -> Modify this to filter for tokens where the teacher's focus and the model's generated token both fall within the top 10.
        # probs = outputs['logits'][range(outputs["logits"].shape[0]), continue_ids].sum().item()
        # # ---------------------------------------------------------------------------------------------------------------
        external_similarity = []  # 这个用来存储生成的 token embedding 和 copy head 关注的 token embedding 的相似度得分 -> This is used to store the similarity scores between the generated token embeddings and the token embeddings attended to by the copy head.
        parameter_knowledge_difference = []
        hallucination_label = []
        # 计算一下输入的 context 里面有没有 hallucination 词，如果有的话 copy 的时候把他们的 pointer weight 调小 -> Check the input context for "hallucination" words; if any are found, reduce their pointer weights during copying.
        # input: input_ids, corr token vocab distribution
        # output: hallucination score for the input_ids or hallucination mask
        # outputs.attentions is a tuple, taking the last layer's attentions
        # TODO: make sure attn_layer_id, head_id exist in model prior to iterating
        attentions_list: List[torch.Tensor] = [outputs.attentions[attn_layer_id][:, head_id, :, :] for
                                               attn_layer_id, head_id in copy_heads]
        # Step 1: Average the attention across the number of heads
        for seq_i in range(prefix_ids.shape[-1] - 1, input_ids.shape[-1] - 1):
            torch.cuda.empty_cache()
            # Step 2: Extract the non-zero values from the last row/column
            # Now we gather the attention scores for the last token of each sequence
            pointer_scores_list: list[torch.Tensor] = [attention[:, seq_i, :] for attention in
                                                       attentions_list]  # shape: (batch_size, sequence_length)
            # Step 3: Perform a softmax over the modified attention scores
            # pointer_probs = nn.F.softmax(pointer_scores, dim=-1)  # shape: (batch_size, sequence_length)
            pointer_probs_list = torch.cat(
                [pointer_scores[:, :prefix_ids.shape[-1]] for pointer_scores in pointer_scores_list],
                dim=0)  # shape: (batch_size, prefix_sequence_length) 截取这一步还是只让模型关注文本内容

            # Step 4: select the top attented token
            # Create an extended attention mask that masks out special tokens
            # hyperparameter: token rate

            # pointer_probs_list 是每个位置对应的大小(head_num, seq_len)，last_hidden_states shape (seq_len, hidden_state)是每个位置对应的 value，请取出 top 10% input_ids_cp 的 last_hidden_states，最终输出为(head_num, top10_len, hidden_state)
            # 获取top 10%的索引
            # ->
            # `pointer_probs_list` contains values ​​of shape `(head_num, seq_len)` for each position, and
            # `last_hidden_states` (with shape `(seq_len, hidden_state)`) represents the values ​​corresponding
            # to each position. Please extract the `last_hidden_states` corresponding to the top 10% of `input_ids_cp`
            # (based on the indices of the top 10%) to produce a final output with the shape `(head_num, top10_len, hidden_state)`.
            top_k = int(pointer_probs_list.shape[-1] * 0.1)  # 10% of sequence length

            # 获取排序后的索引，按照概率从大到小排序 -> Obtain the sorted indices, ordered by probability from highest to lowest.
            sorted_indices = torch.argsort(pointer_probs_list, dim=1, descending=True)

            # 选择前top_k个索引 -> Select the top-k indices.
            top_k_indices = sorted_indices[:, :top_k]

            # 我们需要将 top_k_indices 展平，以便用于索引 last_hidden_states -> We need to flatten `top_k_indices` so that it can be used to index `last_hidden_states`.
            flattened_indices = top_k_indices.flatten()  # shape (head_num * k,)
            # 使用展平的索引在 last_hidden_states 中查找相应的 hidden_state -> Use the flattened indices to look up the corresponding hidden state in `last_hidden_states`.
            selected_hidden_states = last_hidden_states[flattened_indices]  # shape (head_num * k, hidden_state)
            # 重新 reshape 成 (head_num, k, hidden_state) -> Reshape into (head_num, k, hidden_state)
            top_k_hidden_states = selected_hidden_states.view(top_k_indices.shape[0], top_k_indices.shape[1], -1)

            attend_token_hidden_state = torch.mean(top_k_hidden_states, dim=1)  # (head_num, hidden_state)

            # Step 5: Calculate the similarity between the last token and the attentioned prefix text
            current_hidden_state = last_hidden_states[seq_i, :]  # shape (hidden_state,)

            # 扩展 current_hidden_state 的形状以匹配 pointer_probs_list -> Expand the shape of current_hidden_state to match pointer_probs_list.
            current_hidden_state = current_hidden_state.unsqueeze(0).expand(attend_token_hidden_state.shape)
            # 计算余弦相似度 -> Calculate cosine similarity.
            cosine_similarity = F.cosine_similarity(attend_token_hidden_state.to(model.device),
                                                    current_hidden_state.to(model.device), dim=1)
            hallucination_label.append(is_hallucination_token(seq_i, hallucination_spans))
            external_similarity.append(cosine_similarity)
            parameter_knowledge_difference.append(
                [calculate_dist(value[0][0, seq_i, :], value[1][0, seq_i, :]) for value in logits_dict.values()])

        dc[dataset_key] = {
            "key": dataset_key,
            "external_similarity": external_similarity,
            "parameter_knowledge_difference": parameter_knowledge_difference,
            "hallucination_label": hallucination_label,
            **dataset_value
        }
        torch.cuda.empty_cache()
        gc.collect()

    dc["info"] = {"copy_heads": copy_heads}
    return dc


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='ReDeEP token level detection.')
    parser.add_argument("-m", '--model_name', type=str, required=True, help='huggingface model identifyer')
    parser.add_argument("-d", "--dataset_path", type=str, required=True, help=f"path to dataset")
    parser.add_argument("-c", "--copy_heads_path", type=str, required=False, default=None,
                        help="topk heads to use as json_file.")  # TODO: impl 'all' flag
    parser.add_argument("-o", "--output", type=str, default="./redeep_token_level_detection.json",
                        help="output path. Default: ./redeep_token_level_detection.json")
    parser.add_argument("--cache_dir", type=str, default="./cache_dir",
                        help="cache directory for saving superficial data")
    parser.add_argument("-t", "--token", type=str, help="huggingface token. can also be set using environmental.")
    parser.add_argument("-a", "--amount", type=int, default=-1, help="amount of datapoints to analyze")
    parser.add_argument("-k", "--knowledge_layers", required=False, nargs=2, default=[0, 32], help="knowledge layers")

    args = parser.parse_args()
    args.knowledge_layers = [int(i) for i in args.knowledge_layers]
    return args


def main(args: argparse.Namespace):
    """Main function to orchestrate the processing pipeline."""
    # setup
    copy_heads, copy_heads_model = load_copy_heads(args.copy_heads_path)
    if args.model_name != copy_heads_model:
        warnings.warn(
            f"provided copy_heads file was created with different model as currently provided. Please check that this is expected. model_name={args.model_name} copy_heads_model={copy_heads_model}")

    dataset = load_data(args.dataset_path, args.amount)
    model, tokenizer = load_model_and_tokenizer(args.model_name, args.cache_dir, args.token)

    # redeep
    processed_responses: Dict[str, Dict[str, Any]] = process_responses(dataset, model, tokenizer, copy_heads,
                                                                       args.knowledge_layers)
    # saving
    save_path = Path(args.output)
    with save_path.open("w") as f:
        json.dump(processed_responses, f, ensure_ascii=False, cls=JsonEncoder, indent=1, )
    print(f"Results saved to {save_path}")


def test_args():
    args = argparse.Namespace()
    args.model_name = "meta-llama/Llama-2-7b-chat-hf"
    args.dataset_path = "./dataset/response_span_llama-2-7b-chat.json"
    args.copy_heads_path = "./copy_heads/llama27b_copy_heads.json"
    args.token = ""
    args.output = "./test_output.json"
    args.cache_dir = "./.cache_dir"
    args.amount = 5
    args.knowledge_layers = [0, 32]
    return args


if __name__ == "__main__":
    args = parse_arguments()
    # args = test_args()
    main(args)
