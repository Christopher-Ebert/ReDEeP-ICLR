from enum import Enum
from dataclasses import dataclass


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
        ragtruth: dict
        dolly: dict
        ragtruth_chunk: dict
        dolly_chunk: dict

    MODEL_CONFIGS = {
        ModelName.LLAMA2_7B: ModelConfig(
            name=ModelName.LLAMA2_7B.value,
            topk_heads_path="./log/test_llama2_7B/topk_heads.json",
            start_layer=0,
            num_layers=32,
            ragtruth={"top_n_heads": 1, "top_n_layers": 10, "external_sim_scaling": 0.2, "param_know_scaling": 1},
            # original naming convention top_n, top_k, alpha, m
            dolly={"top_n_heads": 4, "top_n_layers": 3, "external_sim_scaling": 0.2, "param_know_scaling": 1},
            ragtruth_chunk={"top_n_heads": 3, "top_n_layers": 4, "external_sim_scaling": 0.6, "param_know_scaling": 1},
            dolly_chunk={"top_n_heads": 7, "top_n_layers": 3, "external_sim_scaling": 1.2, "param_know_scaling": 1},
        ),
        ModelName.LLAMA2_13B: ModelConfig(
            name=ModelName.LLAMA2_13B.value,
            topk_heads_path="./log/test_llama2_13B/topk_heads.json",
            start_layer=8,
            num_layers=40,
            ragtruth={"top_n_heads": 2, "top_n_layers": 17, "external_sim_scaling": 0.6, "param_know_scaling": 1},
            dolly={"top_n_heads": 4, "top_n_layers": 5, "external_sim_scaling": 0.6, "param_know_scaling": 1},
            ragtruth_chunk={"top_n_heads": 9, "top_n_layers": 3, "external_sim_scaling": 1.8, "param_know_scaling": 1},
            dolly_chunk={"top_n_heads": 11, "top_n_layers": 3, "external_sim_scaling": 0.2, "param_know_scaling": 1},
        ),
        ModelName.LLAMA3_8B: ModelConfig(
            name=ModelName.LLAMA3_8B.value,
            topk_heads_path="./log/test_llama3_8B/topk_heads.json",
            start_layer=0,
            num_layers=16,
            ragtruth={"top_n_heads": 3, "top_n_layers": 30, "external_sim_scaling": 0.4, "param_know_scaling": 1},
            dolly={"top_n_heads": 1, "top_n_layers": 1, "external_sim_scaling": 0.1, "param_know_scaling": 1},
            ragtruth_chunk={"top_n_heads": 2, "top_n_layers": 5, "external_sim_scaling": 1.2, "param_know_scaling": 1},
            dolly_chunk={"top_n_heads": 1, "top_n_layers": 1, "external_sim_scaling": 0.1, "param_know_scaling": 1},
        ),
    }

    class Dataset(Enum):
        """Enumeration of supported datasets."""
        RAGTRUTH = "ragtruth"
        DOLLY = "dolly"

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
