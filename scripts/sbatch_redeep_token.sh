#!/bin/bash

hf_token=$1

submit_step () {
    local command=$1
    local ident=$2
    local token=$3
    sbatch --parsable \
<< EOF
#!/bin/bash
#SBATCH --job-name=ReDEeP_${ident}
#SBATCH --output=logs/%j
#SBATCH --error=logs/%j
#SBATCH --time=01-00:00:00
#SBATCH --mem=100gb
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --partition=A100-80GB,H100,H100-SLT,H100-PCI,H200
#SBATCH --container-image=/enroot/nvcr.io_nvidia_pytorch_26.06-py3.sqsh
#SBATCH --container-mounts=/netscratch/ebert/ReDEeP-ICLR:/workspace


mkdir -p ./results
export HF_TOKEN=$token

pip install datasets numpy==2.5.1 accelerate sentence-transformers tokenizers tqdm
pip install --upgrade ./transformers-4.42.0.dev0-py3-none-any.whl
hf auth login --token "\$HF_TOKEN"
python "$command"
EOF
}


DATASETS=(
'./dataset/response.jsonl'
'./dataset/response_span_llama-2-7b-chat.json'
'./dataset/response_spans.jsonl'
'./dataset/response_span_llama-2-70b-chat.json'
'./dataset/response_span_gpt-4-0613.json'
'./dataset/response_span_llama-2-13b-chat.json'
'./dataset/response_with_llama3_8b_spans.jsonl'
'./dataset/source_info.jsonl'
'./dataset/response_with_llama3_8b.jsonl'
'./dataset/response_span_mistral-7B-instruct.json'
'./dataset/source_info_spans.jsonl'
'./dataset/response_span_gpt-3.5-turbo-0613.json'
'./data/ragbench_covidqa_test.json'
'./data/ragbench_hotpotqa_test.json'
'./data/ragbench_msmarco_test.json'
'./data/ragbench_pubmedqa_test.json'
'./data/ragonize-Llama-2-7b-chat-hf-test.json'
'./data/ragonize-Llama-2-7b-chat-hf-train.json'
'./data/ragonize-Llama-3.1-8B-Instruct-test.json'
'./data/ragonize-Llama-3.1-8B-Instruct-train.json'
'./data/ragonize-Mistral-7B-Instruct-v0.1-test.json'
'./data/ragonize-Mistral-7B-Instruct-v0.1-train.json'
'./data/ragonize-Mistral-7B-Instruct-v0.3-test.json'
'./data/ragonize-Mistral-7B-Instruct-v0.3-train.json'
'./data/ragtruth-llama-2-7b-chat-test.json'
'./data/ragtruth-llama-2-7b-chat-train.json'
'./data/ragtruth-mistral-7B-instruct-test.json'
'./data/ragtruth-mistral-7B-instruct-train.json'
)

DATASET_IDENTS=(
'response'
'response_span_llama-2-7b-chat'
'response_spans'
'response_span_llama-2-70b-chat'
'response_span_gpt-4-0613'
'response_span_llama-2-13b-chat'
'response_with_llama3_8b_spans'
'source_info'
'response_with_llama3_8b'
'response_span_mistral-7B-instruct'
'source_info_spans'
'response_span_gpt-3.5-turbo-0613'
'ragbench_covidqa_test'
'ragbench_hotpotqa_test'
'ragbench_msmarco_test'
'ragbench_pubmedqa_test'
'ragonize-Llama-2-7b-chat-hf-test'
'ragonize-Llama-2-7b-chat-hf-train'
'ragonize-Llama-3.1-8B-Instruct-test'
'ragonize-Llama-3.1-8B-Instruct-train'
'ragonize-Mistral-7B-Instruct-v0.1-test'
'ragonize-Mistral-7B-Instruct-v0.1-train'
'ragonize-Mistral-7B-Instruct-v0.3-test'
'ragonize-Mistral-7B-Instruct-v0.3-train'
'ragtruth-llama-2-7b-chat-test'
'ragtruth-llama-2-7b-chat-train'
'ragtruth-mistral-7B-instruct-test'
'ragtruth-mistral-7B-instruct-train'
)

MODELS=(
'meta-llama/Llama-2-7b-chat-hf'
'meta-llama/Llama-2-13b-chat-hf'
'meta-llama/Meta-Llama-3-8B-Instruct'
)

MODEL_IDENTS=(
'Llama-2-7b-chat-hf'
'Llama-2-13b-chat-hf'
'Meta-Llama-3-8B-Instruct'
)

COPYHEADS=(
'./copy_heads/llama27b_copy_heads.json'
'./copy_heads/llama213b_copy_heads.json'
'./copy_heads/llama38b_copy_heads.json'
)

HOT_VALUES=(
'-nh 1 -nl 10 -es 0.2 -pk 1'
'-nh 2 -nl 17 -es 0.6 -pk 1'
'-nh 3 -nl 30 -es 0.4 -pk 1'
)

for i in "${!MODELS[@]}"; do
  model="${MODELS[$i]}"
  model_ident="${MODEL_IDENTS[$i]}"
  copy_head="${COPYHEADS[$i]}"
  hot_val="${HOT_VALUES[$i]}"

  for q in "${!DATASETS[@]}"; do
    dataset="${DATASETS[$q]}"
    dataset_ident="${DATASET_IDENTS[$q]}"
    command="token_level_rewrite.py -m $model -d $dataset -c $copy_head -t $hf_token -o ./results/${dataset_ident}_${model_ident}.json ${hot_val}"
    batch_id=$(submit_step "$command" "${dataset_ident}_${model_ident}" "$hf_token")
    echo "$batch_id"
  done
done