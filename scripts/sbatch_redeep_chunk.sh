#!/bin/bash
#SBATCH --job-name=redeep_experiment
#SBATCH --output=logs/%j
#SBATCH --error=logs/%j
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=100G
#SBATCH --gpus=1
#SBATCH --time=10:00:00
#SBATCH --partition=A100-80GB,H100,H100-SLT,H200
#SBATCH --container-mounts=/netscratch/ebert/ReDEeP-ICLR:/workspace
#SBATCH --container-image=/enroot/nvcr.io_nvidia_pytorch_25.05-py3.sqsh

hf_token=$1

./mkenv.sh
python chunk_level_detect_rewrite.py -m meta-llama/Llama-2-7b-chat-hf -d ./dataset/response_span_llama-2-7b-chat.json -c ./copy_heads/llama27b_copy_heads.json -t "$hf_token" -o ./redeep_llama27b_chunk.json
python chunk_level_reg_rewrite.py -d ./redeep_llama27b.json -nh 1 -nl 10 -a 0.2 -m 1 -o ./redeep_llama27b_reg_score_chunk.json