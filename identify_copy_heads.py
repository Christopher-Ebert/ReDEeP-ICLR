"""
Fast Copying Head Identification using Trace + Gershgorin Approximation. Based upon description in ReDeep. This is explicitly written for
llama models.

Instead of explicitly computing eigenvalues of M (which is expensive), descriped in 'Elhage et al. A mathematical framework for
transformer circuits. Transformer Circuits Thread, 2021. URL https://transformer-circuits.pub/2021/framework/index.html'
we estimate Copying Head behavior using:

1. Trace(M), where M = W_U @ W_OV @ W_E
2. Gershgorin Circle Theorem
3. IQR-Based Outlier Detection for Gershgorin boundary Points
4. Copying Head Score Calculation

The final Copying Head Score is
score = rank(outliers ascending) + rank(|trace| descending)
Smaller scores indicate stronger Copying Head candidates.
"""
import argparse
from dataclasses import dataclass
from typing import List

import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


@dataclass
class HeadScore:
    layer: int
    head: int
    trace: float
    abs_trace: float
    outliers: int
    trace_rank: int = 0
    outlier_rank: int = 0
    copying_score: int = 0


def build_matrix(W_U: np.ndarray, W_OV: np.ndarray, W_E: np.ndarray) -> np.ndarray:
    """
    Constructs M = W_U @ W_OV @ W_E
    :param W_E: (d_model, vocab)
    :param W_OV: (d_model, d_model)
    :param W_U: (vocab, d_model)
    :return: (vocab, vocab)
    """
    return W_U @ W_OV @ W_E


def gershgorin_boundary_points(M: np.ndarray) -> np.ndarray:
    """
    Compute the left/right boundary of every Gershgorin disk.

    Disk:
        center = a_ii
        radius = sum_j!=i |a_ij|

    Boundary points:
        center - radius
        center + radius
    """
    diag = np.diag(M)
    abs_rows = np.sum(np.abs(M), axis=1)
    radius = abs_rows - np.abs(diag)
    left = diag - radius
    right = diag + radius
    return np.concatenate([left, right])


def count_iqr_outliers(values: np.ndarray):
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = np.sum((values < lower) | (values > upper))

    return int(outliers), lower, upper


# single head analysis
def analyze_head(layer: int, head: int, M: np.ndarray) -> HeadScore:
    """

    :param layer:
    :param head:
    :param W_E:
    :param W_OV:
    :param W_U:
    :return:
    """
    tr = float(np.trace(M))
    boundary = gershgorin_boundary_points(M)
    outliers, _, _ = count_iqr_outliers(boundary)
    return HeadScore(
        layer=layer,
        head=head,
        trace=tr,
        abs_trace=abs(tr),
        outliers=outliers,
    )


# copy head score calculation
def compute_copying_scores(scores: List[HeadScore]) -> List[HeadScore]:
    """
    Ranks heads according to Outlier count (ascending) and |Trace| descending. copy_score = rank_outlier + rank_trace. Lower score
    correspond to stronger copy-head-behavior.
    :param scores:
    :return:
    """
    outlier_order = sorted(scores, key=lambda s: s.outliers)
    trace_order = sorted(scores, key=lambda s: s.abs_trace, reverse=True)

    for r, s in enumerate(outlier_order, start=1):
        s.outlier_rank = r

    for r, s in enumerate(trace_order, start=1):
        s.trace_rank = r

    for s in scores:
        s.copying_score = s.trace_rank + s.outlier_rank

    return sorted(scores, key=lambda s: s.copying_score)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='ReDeEP Copy Head detection.')
    parser.add_argument("-m", '--model_name', type=str, required=True, help='huggingface model identifyer')
    parser.add_argument("-o", "--output", type=str, default="./redeep_copy_heads.json",
                        help="output path. Default: ./redeep_copy_heads.json")
    parser.add_argument("--cache_dir", type=str, default="./cache_dir", help="cache directory for saving superficial data")
    parser.add_argument("-t", "--token", type=str, help="huggingface token. can also be set using environmental.")
    parser.add_argument("-k", "--range_layers", required=False, nargs=2, default=[0, -1], help="range of layers to be analyzed")

    args = parser.parse_args()
    args.range_layers = [int(i) for i in args.range_layers]
    return args


def load_model_and_tokenizer(model_name, cache_dir, hf_token) -> tuple:
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


def main():
    model_name = "meta-llama/Llama-2-7b-hf"
    model, tokenizer = load_model_and_tokenizer(model_name, cache_dir="./.cache_dir", hf_token="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")

    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads

    W_E = model.model.embed_tokens.weight.T  # (d_model, vocab)
    copy_heads = []
    for layer_num in range(num_layers):
        layer = model.model.layers[layer_num]
        for head_num in range(num_heads):
            start_idx = head_num * head_dim
            end_idx = (head_num + 1) * head_dim
            W_U = layer.self_attn.q_proj.weight[:, start_idx:end_idx]
            W_O = layer.self_attn.o_proj.weight[:, start_idx:end_idx]  # d_model × d_head
            W_V = layer.self_attn.v_proj.weight[start_idx:end_idx, :]  # d_head × d_model
            W_OV = W_O @ W_V  # d_model × d_model
            M = build_matrix(W_U, W_OV, W_E)
            copy_heads.append(analyze_head(layer_num, head_num, M))
    copy_heads = compute_copying_scores(copy_heads)
    return copy_heads


if __name__ == "__main__":
    # args = parse_arguments()
    main()
