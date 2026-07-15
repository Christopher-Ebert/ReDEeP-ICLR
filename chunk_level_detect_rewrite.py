from enum import Enum
from dataclasses import dataclass
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
import os
from itertools import pairwise
from torch import randn

cache_folder = "./data"
os.environ["HF_TOKEN"] = ""


class ReDeEP_Configs:
    class ModelName(Enum):
        """Enumeration of supported model names."""
        LLAMA2_7B = "meta-llama/Llama-2-7b-chat-hf"
        LLAMA2_13B = "meta-llama/Llama-2-13b-chat-hf"
        LLAMA3_8B = "meta-llama/Meta-Llama-3-8B-Instruct"

    @dataclass
    class ModelConfig:
        """Configuration for a specific model."""
        name: str
        topk_heads_path: str
        start_layer: int
        num_layers: int

    # Model configurations mapping
    MODEL_CONFIGS = {
        ModelName.LLAMA2_7B: ModelConfig(
            name="llama2/llama-2-7b-chat-hf",
            topk_heads_path="./log/test_llama2_7B/topk_heads.json",
            start_layer=0,
            num_layers=32
        ),
        ModelName.LLAMA2_13B: ModelConfig(
            name="llama2/llama-2-13b-chat-hf",
            topk_heads_path="./log/test_llama2_13B/topk_heads.json",

            start_layer=8,
            num_layers=40
        ),
        ModelName.LLAMA3_8B: ModelConfig(
            name="llama3/Meta-Llama-3-8B-Instruct/",
            topk_heads_path="./log/test_llama3_8B/topk_heads.json",

            start_layer=0,
            num_layers=16
        ),
    }

    class Dataset(Enum):
        """Enumeration of supported datasets."""
        RAGTRUTH = "ragtruth"
        DOLLY = "dolly"

    # Dataset paths mapping
    DATASET_PATHS = {
        Dataset.RAGTRUTH: {
            "response_path": "/mnt/internal/sata-ssd/GitHub/SteffenLuminaETC/ReDeEP/dataset/response.jsonl",
            "source_info_path": "/mnt/internal/sata-ssd/GitHub/SteffenLuminaETC/ReDeEP/dataset/source_info_spans.jsonl",
        },
        Dataset.DOLLY: {
            "response_path": "../dataset/response_dolly_spans.jsonl",
            "source_info_path": "../dataset/source_info_dolly_spans.jsonl",
        },
    }


def load_data(dataset_path):
    with Path(dataset_path).open() as f:
        data = json.load(f)
    return data


def load_model_and_tokenizer(model_name):
    """Load the model, tokenizer, and optional tokenizer for template."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        dtype=torch.float16,
        cache_dir=cache_folder,
        attn_implementation="eager"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_folder)
    return model, tokenizer

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
    return sum(scores) if isinstance(list, scores) else scores


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


def is_hallucination_span(r_span: List[int], hallucination_spans: List[List[int]]) -> bool:
    """Check if any token in a response span falls within any hallucination span."""
    for token_id in range(r_span[0], r_span[1]):
        if is_hallucination_token(token_id, hallucination_spans):
            return True
    return False


def calculate_hallucination_spans(
        response: List[Dict],
        text: str,
        response_rag: str,
        tokenizer: Any,
        prefix_len: int
) -> List[List[int]]:
    """Calculate hallucination spans in token IDs."""
    hallucination_span = []
    for item in response:
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

def calculate_respond_spans(
        raw_response_spans: List[List[int]],
        text: str,
        response_rag: str,
        tokenizer: Any
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


def calculate_sentence_similarity(r_text: str, p_text: str, bge_model: SentenceTransformer) -> float:
    """Calculate sentence similarity using BGE model."""
    part_embedding = bge_model.encode([r_text], normalize_embeddings=True)
    q_embeddings = bge_model.encode([p_text], normalize_embeddings=True)

    scores_named = np.matmul(q_embeddings, part_embedding.T).flatten()
    return float(scores_named[0])

def process_responses(
        dataset,
        model: Any,
        tokenizer: Any,
        copy_heads: Iterable[Iterable[int]],
        bge_model: SentenceTransformer,
) -> Dict:
    """Process a single response item and calculate scores."""
    dc = {}
    for k, v in dataset.items():
        response_rag = v['response']
        prompt = v['prompt']
        prompt_spans = v["prompt_spans"]
        original_prompt_spans = v['prompt_spans']
        original_response_spans = v['response_spans']
        labels: List | Tuple = v["labels"]

        text = add_special_template(prompt[:12000], tokenizer)
        input_text = text + response_rag

        print("all_text_len:", len(input_text))
        print("prompt_len", len(prompt))
        print("respond_len", len(response_rag))

        input_ids = tokenizer([input_text], return_tensors="pt").input_ids
        prefix_ids = tokenizer([text], return_tensors="pt").input_ids
        continue_ids = input_ids[0, prefix_ids.shape[-1]:] # not used in original code base as well

        hallucination_spans = []
        if labels is not None or len(labels) != 0:
            hallucination_spans = calculate_hallucination_spans(
                labels, text, response_rag, tokenizer, prefix_ids.shape[-1]
            )

        prompt_spans = calculate_prompt_spans(prompt_spans, prompt, tokenizer,)
        respond_spans = calculate_respond_spans(original_response_spans, text, response_rag, tokenizer)

        with torch.no_grad():
            outputs = model(  # originally: logits_dict, outputs
                input_ids=input_ids,
                return_dict=True,
                output_attentions=True,
                output_hidden_states=True,
                # knowledge_layers=list(range(model_config.start_layer, model_config.num_layers)) #TODO: do this
            )
        # not sure what this does, probably pushing the labels (non-hallucinated:0, hallucinated:1) layerwise to device?
        # logits_dict = {key: [value[0].to(model.device), value[1].to(model.device)] for key, value in logits_dict.items()}

        # pairwise combining each layers output with the next.
        logits_pairwise = pairwise(outputs.logits[0])
        for response_id, response_span in enumerate(respond_spans):
            layer_head_span = {}
            # assumes that attn_layer and head exist in model
            for attn_layer_id, head_id in copy_heads:
                scores = [] # p_span_score_dict. only saving mapping score, not p_span
                #Step 1, Eq.2 identify attended tokens
                for prompt_span in prompt_spans:
                    attention_score = outputs.attentions[attn_layer_id][0, head_id, :, :]
                    _score = torch.sum(attention_score[response_span[0]:response_span[1], prompt_span[0]:prompt_span[1]]).cpu().item()
                    scores.append(_score)
                p_id = scores.index(max(scores)) # prompt_spans[scores.index(max(scores))] # Extrahieren Sie das p_span, das dem höchsten Wert entspricht.
                prompt_span_text = prompt[original_prompt_spans[p_id][0]:original_prompt_spans[p_id][1]]
                respond_span_text = response_rag[original_response_spans[response_id][0]:original_response_spans[response_id][1]]
                layer_head_span[str((attn_layer_id, head_id))] = calculate_sentence_similarity(prompt_span_text, respond_span_text, bge_model)

            parameter_knowledge_scores = [
                calculate_dist_2d(value[0][response_span[0]:response_span[1]], value[1][response_span[0]:response_span[1]])
                for value in list(logits_pairwise) #logits_dict.values()
            ]
            parameter_knowledge_dict = {f"layer_{i}": value for i, value in enumerate(parameter_knowledge_scores)}

            dc[k] = {
                "key": k,
                "prompt_attention_score": layer_head_span,
                "response_span": response_span,
                "hallucination_label": 1 if is_hallucination_span(response_span, hallucination_spans) else 0,
                "parameter_knowledge_scores": parameter_knowledge_dict
            }
    return dc


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='ReDeEP Chunk detection.')
    parser.add_argument(
        '--model_name',
        type=str,
        required=True, help='huggingface model'
    )
    parser.add_argument("--response_path")
    parser.add_argument("--source_info_path")
    parser.add_argument("--topk_heads_path", type=str, help="topk heads to use")
    parser.add_argument("--output", type=str, default=..., help="output path")  # TODO: do me.
    return parser.parse_args()


def main():
    """Main function to orchestrate the processing pipeline."""
    # args = parse_arguments()
    args = argparse.Namespace()
    # TODO remove test params
    args.model_name = ReDeEP_Configs.ModelName.LLAMA2_7B.value
    args.dataset_path = r"/mnt/internal/sata-ssd/GitHub/SteffenLuminaETC/ReDeEP/dataset/response_span_llama-2-7b-chat.json"
    copy_heads = [[25, 0], [18, 13], [18, 10], [27, 9], [5, 29], [23, 8], [31, 28], [3, 0], [31, 24], [13, 20],
                  [31, 18], [1, 14], [2, 5], [22, 10], [2, 22], [15, 7], [3, 19], [20, 17], [10, 20], [23, 30],
                  [20, 22], [1, 27], [20, 1], [31, 19], [28, 18], [20, 15], [1, 21], [19, 1], [20, 5], [16, 1], [18, 9],
                  [5, 13]]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = args.model_name

    bge_model = SentenceTransformer("BAAI/bge-base-en-v1.5", cache_folder=cache_folder).to(device)
    dataset = load_data(args.dataset_path)
    model, tokenizer = load_model_and_tokenizer(model_name)
    processed_responses = process_responses(
        dataset,
        model,
        tokenizer,
        copy_heads,
        bge_model,
    )
    exit()
    save_path = Path("./test_save_path.json")
    with save_path.open("w") as f:
        json.dump(processed_responses, f, ensure_ascii=False)

    print(f"Results saved to {save_path}")


if __name__ == "__main__":
    main()
