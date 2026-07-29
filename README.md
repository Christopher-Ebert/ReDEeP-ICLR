# ReDeEP Replication

This repository is a replication of the ReDeEP project.

For the original repository, please visit: [https://github.com/Jeryi-Sun/ReDEeP-ICLR](#)

## Usage

To use this replication, follow these steps:

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the token-level detection script:
   ```bash
   python token_level_detect.py \
     -m meta-llama/Llama-2-7b-chat-hf \
     -d ./dataset/response_span_llama-2-7b-chat.json \
     -c ./copy_heads/llama27b_copy_heads.json \
     -t <token> \
     -o ./token_level_detect_llama27b.json
   ```

3. Run the token-level rewrite script:
   ```bash
   python token_level_reg_rewrite.py \
     -d ./token_level_detect_llama27b.json \
     -nh 1 \
     -nl 10 \
     -a 0.2 \
     -m 1 \
     -o ./token_level_req_llama27b.json
   ```

### Token-Level Detection Script Arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `-m`, `--model_name` | str | Yes | - | Huggingface model identifier (e.g., "meta-llama/Llama-2-7b-chat-hf") |
| `-d`, `--dataset_path` | str | Yes | - | Path to the dataset file in JSON format |
| `-c`, `--copy_heads_path` | str | No | None | Path to JSON file containing top-k heads to use |
| `-o`, `--output` | str | No | "./redeep_token_level_detection.json" | Output file path for results |
| `--cache_dir` | str | No | "./cache_dir" | Cache directory for saving model data |
| `-t`, `--token` | str | No | - | Huggingface token (can also be set via environment variable) |
| `-a`, `--amount` | int | No | -1 | Number of datapoints to analyze (-1 for all) |
| `-k`, `--knowledge_layers` | int[] | No | [0, 32] | Range of knowledge layers to use (two values: start and end) |

### Token-Level Regression Rewrite Script Arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `-d`, `--dataset_path` | str | Yes | - | Path to the token-level detection output file in JSON format |
| `-o`, `--output` | str | No | "./redeep_token_level_reg.json" | Output file path for regression results |
| `-nh`, `--top_n_heads` | int | No | 1 | Number of top attention heads to select based on AUC |
| `-nl`, `--top_n_layers` | int | No | 10 | Number of top layers to select based on AUC |
| `-a`, `--external_sim_scaling` | float | No | 0.2 | Scaling factor for external similarity features |
| `-m`, `--param_know_scaling` | int | No | 1 | Scaling factor for parameter knowledge difference features |

The `chunk_level_detect_rewrite.py` and `chunk_level_reg_rewrite.py` scripts accept the same arguments as their token-level counterparts, with equivalent functionality at the chunk level.

### Performance Comparison

| Metric | Original Paper | Original Repository | This Replication |
|--------|----------------|---------------------|------------------|
| AUC    | -              | -                   | -                |
| PCC    | -              | -                   | -                |
