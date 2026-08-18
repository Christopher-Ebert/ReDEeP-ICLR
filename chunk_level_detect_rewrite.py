import warnings
from pathlib import Path
from typing import List, Tuple, Dict, Any, Iterable
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from torch.nn import functional as F
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import numpy as np
import argparse


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
                knowledge_layers=list(range(knowledge_layers[0], knowledge_layers[1]))
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


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='ReDeEP token level detection.')
    parser.add_argument("-m", '--model_name', type=str, required=True, help='huggingface model identifyer')
    parser.add_argument("-d", "--dataset_path", type=str, required=True, help=f"path to dataset")
    parser.add_argument("-c", "--copy_heads_path", type=str, required=False, default=None,
                        help="topk heads to use as json_file.")
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
            f"provided copy_heads file was created with different model as currently provided. Please check that this is expected. "
            f"model_name={args.model_name} copy_heads_model={copy_heads_model}")

    dataset = load_data(args.dataset_path, args.amount)
    model, tokenizer = load_model_and_tokenizer(args.model_name, args.cache_dir, args.token)

    bge_model = SentenceTransformer("BAAI/bge-base-en-v1.5", cache_folder=args.cache_dir, token=args.token).to("cuda")
    processed_responses = process_responses(
        dataset,
        model,
        tokenizer,
        copy_heads,
        bge_model,
        args.knowledge_layers
    )
    # saving
    save_path = Path(args.output).absolute()
    with save_path.open("w") as f:
        json.dump(processed_responses, f, ensure_ascii=False, cls=JsonEncoder, indent=1, )
    print(f"Results saved to {save_path}")


def test_args():
    args = argparse.Namespace()
    args.model_name = "meta-llama/Llama-2-7b-chat-hf"
    args.dataset_path = "./dataset/response_span_llama-2-7b-chat.json"
    args.copy_heads_path = "./copy_heads/llama27b_copy_heads.json"
    args.token = ""
    args.output = "./test_output_chunk.json"
    args.cache_dir = "./.cache_dir"
    args.amount = 5
    args.knowledge_layers = [0, 32]
    return args


if __name__ == "__main__":
    args = parse_arguments()
    # args = test_args()
    main(args)
