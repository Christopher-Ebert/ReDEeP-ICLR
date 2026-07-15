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

    # Model configurations mapping
    MODEL_CONFIGS = {
        ModelName.LLAMA2_7B: ModelConfig(
            name=ModelName.LLAMA2_7B.value,
            topk_heads_path="./log/test_llama2_7B/topk_heads.json",
            start_layer=0,
            num_layers=32,
            ragtruth={"top_n":1, "top_k":10, "alpha":0.2, "m":1},
            dolly={"top_n":4, "top_k":3, "alpha":0.2, "m":1}
        ),
        ModelName.LLAMA2_13B: ModelConfig(
            name=ModelName.LLAMA2_13B.value,
            topk_heads_path="./log/test_llama2_13B/topk_heads.json",
            start_layer=8,
            num_layers=40,
            ragtruth={"top_n": 2, "top_k": 17, "alpha": 0.6, "m": 1},
            dolly={"top_n": 4, "top_k": 5, "alpha": 0.6, "m": 1}
        ),
        ModelName.LLAMA3_8B: ModelConfig(
            name=ModelName.LLAMA3_8B.value,
            topk_heads_path="./log/test_llama3_8B/topk_heads.json",
            start_layer=0,
            num_layers=16,
            ragtruth={"top_n": 3, "top_k": 30, "alpha": 0.4, "m": 1},
            dolly={"top_n": 1, "top_k": 1, "alpha": 0.1, "m": 1}
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