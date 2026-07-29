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

# llama 2 7b
detect_out=./ragtruth_llama2_7b_chunk_detect.json
reg_out=/ragtruth_llama2_7b_chunk_reg.json

python chunk_level_detect_rewrite.py -m meta-llama/Llama-2-7b-chat-hf -d ./dataset/response_span_llama-2-7b-chat.json -c ./copy_heads/llama27b_copy_heads.json -t "$hf_token" -o "$detect_out"
python chunk_level_reg_rewrite.py -d "$detect_out" -nh 3 -nl 4 -a 0.6 -m 1 -o "$reg_out"

# llama 2 13b
detect_out=./ragtruth_llama2_13b_chunk_detect.json
reg_out=/ragtruth_llama2_13b_chunk_reg.json

python chunk_level_detect_rewrite.py -m meta-llama/Llama-2-13b-chat-hf -d ./dataset/response_span_llama-2-13b-chat.json -c ./copy_heads/llama213b_copy_heads.json -t "$hf_token" -o "$detect_out"
python chunk_level_reg_rewrite.py -d "$detect_out" -nh 9 -nl 3 -a 1.8 -m 1 -o "$reg_out"

# llama 3 8b
detect_out=./ragtruth_llama3_8b_chunk_detect.json
reg_out=/ragtruth_llama3_8b_chunk_reg.json

python chunk_level_detect_rewrite.py -m meta-llama/Meta-Llama-3-8B-Instruct -d ./dataset/response_span_llama-3-8b-Instruct.json -c ./copy_heads/llama38b_copy_heads.json -t "$hf_token" -o "$detect_out"
python chunk_level_reg_rewrite.py -d "$detect_out" -nh 2 -nl 5 -a 1.2 -m 1 -o "$reg_out"