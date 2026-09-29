# ReDeEP Replication

This repository is a replication of the ReDeEP project.

For the original repository, please visit: [ReDeEP](https://github.com/Jeryi-Sun/ReDEeP-ICLR)

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

| Argument                   | Type  | Required | Default                               | Description                                                          |
|----------------------------|-------|----------|---------------------------------------|----------------------------------------------------------------------|
| `-m`, `--model_name`       | str   | Yes      | -                                     | Huggingface model identifier (e.g., "meta-llama/Llama-2-7b-chat-hf") |
| `-d`, `--dataset_path`     | str   | Yes      | -                                     | Path to the dataset file in JSON format                              |
| `-c`, `--copy_heads_path`  | str   | No       | None                                  | Path to JSON file containing top-k heads to use                      |
| `-o`, `--output`           | str   | No       | "./redeep_token_level_detection.json" | Output file path for results                                         |
| `--cache_dir`              | str   | No       | "./cache_dir"                         | Cache directory for saving model data                                |
| `-t`, `--token`            | str   | No       | -                                     | Huggingface token (can also be set via environment variable)         |
| `-a`, `--amount`           | int   | No       | -1                                    | Number of datapoints to analyze (-1 for all)                         |
| `-k`, `--knowledge_layers` | int[] | No       | [0, 32]                               | Range of knowledge layers to use (two values: start and end)         |

### Token-Level Regression Rewrite Script Arguments

| Argument                       | Type  | Required | Default                         | Description                                                  |
|--------------------------------|-------|----------|---------------------------------|--------------------------------------------------------------|
| `-d`, `--dataset_path`         | str   | Yes      | -                               | Path to the token-level detection output file in JSON format |
| `-o`, `--output`               | str   | No       | "./redeep_token_level_reg.json" | Output file path for regression results                      |
| `-nh`, `--top_n_heads`         | int   | No       | 1                               | Number of top attention heads to select based on AUC         |
| `-nl`, `--top_n_layers`        | int   | No       | 10                              | Number of top layers to select based on AUC                  |
| `-a`, `--external_sim_scaling` | float | No       | 0.2                             | Scaling factor for external similarity features              |
| `-m`, `--param_know_scaling`   | int   | No       | 1                               | Scaling factor for parameter knowledge difference features   |

The `chunk_level_detect_rewrite.py` and `chunk_level_reg_rewrite.py` scripts accept the same arguments as their
token-level counterparts, with equivalent functionality at the chunk level.

### Performance Comparison

### RAGTruth

##### Token-Level Detection

| Model                               | Dataset                                 | AUC      | PCC       | Top N External Similarity | Top N External Knowledge | Additional Parameters                                                                                                                          |
|-------------------------------------|-----------------------------------------|----------|-----------|---------------------------|--------------------------|------------------------------------------------------------------------------------------------------------------------------------------------|
| meta-llama/Llama-2-13b-chat-hf      | llama2-13b                              | 0.696052 | 0.238832  | 0.620199                  | 0.441942                 | amount=-1, knowledge_layers=[0, 32], top_n_heads=2, top_n_layers=17, external_sim_scaling=0.6, param_know_scaling=1, max_sentence_length=12000 |
| meta-llama/Llama-2-7b-chat-hf       | llama2-13b                              | 0.357286 | -0.170542 | 0.570971                  | 0.389855                 | amount=-1, knowledge_layers=[0, 32], top_n_heads=1, top_n_layers=10, external_sim_scaling=0.2, param_know_scaling=1, max_sentence_length=12000 |
| meta-llama/Meta-Llama-3-8B-Instruct | llama2-13b                              | 0.481020 | -0.038449 | 0.518278                  | 0.439690                 | amount=-1, knowledge_layers=[0, 32], top_n_heads=3, top_n_layers=30, external_sim_scaling=0.4, param_know_scaling=1, max_sentence_length=12000 |
| meta-llama/Llama-2-13b-chat-hf      | llama2-7b                               | 0.544510 | 0.083104  | 0.540739                  | 0.471547                 | amount=-1, knowledge_layers=[0, 32], top_n_heads=2, top_n_layers=17, external_sim_scaling=0.6, param_know_scaling=1, max_sentence_length=12000 |
| meta-llama/Llama-2-7b-chat-hf       | llama2-7b                               | 0.342998 | -0.237374 | 0.536670                  | 0.402187                 | amount=-1, knowledge_layers=[0, 32], top_n_heads=1, top_n_layers=10, external_sim_scaling=0.2, param_know_scaling=1, max_sentence_length=12000 |
| meta-llama/Meta-Llama-3-8B-Instruct | llama2-7b                               | 0.459494 | -0.071089 | 0.469744                  | 0.475851                 | amount=-1, knowledge_layers=[0, 32], top_n_heads=3, top_n_layers=30, external_sim_scaling=0.4, param_know_scaling=1, max_sentence_length=12000 |
| meta-llama/Llama-2-13b-chat-hf      | mistral-7b                              | 0.510115 | 0.004988  | 0.564942                  | 0.463222                 | amount=-1, knowledge_layers=[0, 32], top_n_heads=2, top_n_layers=17, external_sim_scaling=0.6, param_know_scaling=1, max_sentence_length=12000 |
| meta-llama/Llama-2-7b-chat-hf       | mistral-7b                              | 0.336336 | -0.172022 | 0.522166                  | 0.411205                 | amount=-1, knowledge_layers=[0, 32], top_n_heads=1, top_n_layers=10, external_sim_scaling=0.2, param_know_scaling=1, max_sentence_length=12000 |
| meta-llama/Meta-Llama-3-8B-Instruct | mistral-7b                              | 0.401138 | -0.104608 | 0.476457                  | 0.455537                 | amount=-1, knowledge_layers=[0, 32], top_n_heads=3, top_n_layers=30, external_sim_scaling=0.4, param_know_scaling=1, max_sentence_length=12000 |
| meta-llama/Llama-2-13b-chat-hf      | ragbench_covidqa_test                   | 0.544407 | 0.058890  | 0.517051                  | 0.480088                 | amount=-1, knowledge_layers=[0, 32], top_n_heads=2, top_n_layers=17, external_sim_scaling=0.6, param_know_scaling=1, max_sentence_length=12000 |
| meta-llama/Llama-2-7b-chat-hf       | ragbench_covidqa_test                   | 0.509724 | -0.001521 | 0.541589                  | 0.477985                 | amount=-1, knowledge_layers=[0, 32], top_n_heads=1, top_n_layers=10, external_sim_scaling=0.2, param_know_scaling=1, max_sentence_length=12000 |
| meta-llama/Meta-Llama-3-8B-Instruct | ragbench_covidqa_test                   | 0.520129 | 0.022149  | 0.496547                  | 0.507643                 | amount=-1, knowledge_layers=[0, 32], top_n_heads=3, top_n_layers=30, external_sim_scaling=0.4, param_know_scaling=1, max_sentence_length=12000 |
| meta-llama/Llama-2-13b-chat-hf      | ragbench_hotpotqa_test                  | 0.574197 | 0.054095  | 0.551044                  | 0.479570                 | amount=-1, knowledge_layers=[0, 32], top_n_heads=2, top_n_layers=17, external_sim_scaling=0.6, param_know_scaling=1, max_sentence_length=12000 |
| meta-llama/Llama-2-7b-chat-hf       | ragbench_hotpotqa_test                  | 0.483764 | -0.024494 | 0.509019                  | 0.489458                 | amount=-1, knowledge_layers=[0, 32], top_n_heads=1, top_n_layers=10, external_sim_scaling=0.2, param_know_scaling=1, max_sentence_length=12000 |
| meta-llama/Meta-Llama-3-8B-Instruct | ragbench_hotpotqa_test                  | 0.552200 | 0.025431  | 0.505640                  | 0.504339                 | amount=-1, knowledge_layers=[0, 32], top_n_heads=3, top_n_layers=30, external_sim_scaling=0.4, param_know_scaling=1, max_sentence_length=12000 |
| meta-llama/Llama-2-13b-chat-hf      | ragbench_msmarco_test                   | 0.567243 | 0.066291  | 0.515414                  | 0.487208                 | amount=-1, knowledge_layers=[0, 32], top_n_heads=2, top_n_layers=17, external_sim_scaling=0.6, param_know_scaling=1, max_sentence_length=12000 |
| meta-llama/Llama-2-7b-chat-hf       | ragbench_msmarco_test                   | 0.539773 | 0.032158  | 0.510280                  | 0.484924                 | amount=-1, knowledge_layers=[0, 32], top_n_heads=1, top_n_layers=10, external_sim_scaling=0.2, param_know_scaling=1, max_sentence_length=12000 |
| meta-llama/Meta-Llama-3-8B-Instruct | ragbench_msmarco_test                   | 0.529298 | 0.029465  | 0.504020                  | 0.502727                 | amount=-1, knowledge_layers=[0, 32], top_n_heads=3, top_n_layers=30, external_sim_scaling=0.4, param_know_scaling=1, max_sentence_length=12000 |
| meta-llama/Llama-2-13b-chat-hf      | ragbench_pubmedqa_test                  | 0.534050 | 0.043664  | 0.520168                  | 0.481900                 | amount=-1, knowledge_layers=[0, 32], top_n_heads=2, top_n_layers=17, external_sim_scaling=0.6, param_know_scaling=1, max_sentence_length=12000 |
| meta-llama/Llama-2-7b-chat-hf       | ragbench_pubmedqa_test                  | 0.419366 | -0.115350 | 0.517643                  | 0.480402                 | amount=-1, knowledge_layers=[0, 32], top_n_heads=1, top_n_layers=10, external_sim_scaling=0.2, param_know_scaling=1, max_sentence_length=12000 |
| meta-llama/Meta-Llama-3-8B-Instruct | ragbench_pubmedqa_test                  | 0.509849 | 0.022154  | 0.528379                  | 0.476437                 | amount=-1, knowledge_layers=[0, 32], top_n_heads=3, top_n_layers=30, external_sim_scaling=0.4, param_know_scaling=1, max_sentence_length=12000 |
| meta-llama/Llama-2-13b-chat-hf      | ragonize-Llama-2-7b-chat-hf-test        | 0.616857 | 0.193107  | 0.604659                  | 0.452054                 | amount=-1, knowledge_layers=[0, 32], top_n_heads=2, top_n_layers=17, external_sim_scaling=0.6, param_know_scaling=1, max_sentence_length=12000 |
| meta-llama/Llama-2-7b-chat-hf       | ragonize-Llama-2-7b-chat-hf-test        | 0.497602 | -0.012851 | 0.589742                  | 0.440860                 | amount=-1, knowledge_layers=[0, 32], top_n_heads=1, top_n_layers=10, external_sim_scaling=0.2, param_know_scaling=1, max_sentence_length=12000 |
| meta-llama/Meta-Llama-3-8B-Instruct | ragonize-Llama-2-7b-chat-hf-test        | 0.429246 | -0.091422 | 0.473322                  | 0.450405                 | amount=-1, knowledge_layers=[0, 32], top_n_heads=3, top_n_layers=30, external_sim_scaling=0.4, param_know_scaling=1, max_sentence_length=12000 |
| meta-llama/Llama-2-13b-chat-hf      | ragonize-Llama-2-7b-chat-hf-train       | 0.557255 | 0.094624  | 0.546306                  | 0.473515                 | amount=-1, knowledge_layers=[0, 32], top_n_heads=2, top_n_layers=17, external_sim_scaling=0.6, param_know_scaling=1, max_sentence_length=12000 |
| meta-llama/Llama-2-7b-chat-hf       | ragonize-Llama-2-7b-chat-hf-train       | 0.441028 | -0.094326 | 0.539584                  | 0.461342                 | amount=-1, knowledge_layers=[0, 32], top_n_heads=1, top_n_layers=10, external_sim_scaling=0.2, param_know_scaling=1, max_sentence_length=12000 |
| meta-llama/Meta-Llama-3-8B-Instruct | ragonize-Llama-2-7b-chat-hf-train       | 0.455268 | -0.071807 | 0.479306                  | 0.455155                 | amount=-1, knowledge_layers=[0, 32], top_n_heads=3, top_n_layers=30, external_sim_scaling=0.4, param_know_scaling=1, max_sentence_length=12000 |
| meta-llama/Llama-2-13b-chat-hf      | ragonize-Llama-3.1-8B-Instruct-test     | 0.629984 | 0.203074  | 0.616242                  | 0.447576                 | amount=-1, knowledge_layers=[0, 32], top_n_heads=2, top_n_layers=17, external_sim_scaling=0.6, param_know_scaling=1, max_sentence_length=12000 |
| meta-llama/Llama-2-7b-chat-hf       | ragonize-Llama-3.1-8B-Instruct-test     | 0.495227 | -0.014956 | 0.598203                  | 0.436426                 | amount=-1, knowledge_layers=[0, 32], top_n_heads=1, top_n_layers=10, external_sim_scaling=0.2, param_know_scaling=1, max_sentence_length=12000 |
| meta-llama/Meta-Llama-3-8B-Instruct | ragonize-Llama-3.1-8B-Instruct-test     | 0.428975 | -0.085028 | 0.473823                  | 0.448237                 | amount=-1, knowledge_layers=[0, 32], top_n_heads=3, top_n_layers=30, external_sim_scaling=0.4, param_know_scaling=1, max_sentence_length=12000 |
| meta-llama/Llama-2-13b-chat-hf      | ragonize-Llama-3.1-8B-Instruct-train    | 0.646972 | 0.232363  | 0.612758                  | 0.454869                 | amount=-1, knowledge_layers=[0, 32], top_n_heads=2, top_n_layers=17, external_sim_scaling=0.6, param_know_scaling=1, max_sentence_length=12000 |
| meta-llama/Llama-2-7b-chat-hf       | ragonize-Llama-3.1-8B-Instruct-train    | 0.472945 | -0.056272 | 0.593989                  | 0.438938                 | amount=-1, knowledge_layers=[0, 32], top_n_heads=1, top_n_layers=10, external_sim_scaling=0.2, param_know_scaling=1, max_sentence_length=12000 |
| meta-llama/Meta-Llama-3-8B-Instruct | ragonize-Llama-3.1-8B-Instruct-train    | 0.428364 | -0.097956 | 0.477050                  | 0.449345                 | amount=-1, knowledge_layers=[0, 32], top_n_heads=3, top_n_layers=30, external_sim_scaling=0.4, param_know_scaling=1, max_sentence_length=12000 |
| meta-llama/Llama-2-13b-chat-hf      | ragonize-Mistral-7B-Instruct-v0.1-test  | 0.613982 | 0.179062  | 0.610568                  | 0.449710                 | amount=-1, knowledge_layers=[0, 32], top_n_heads=2, top_n_layers=17, external_sim_scaling=0.6, param_know_scaling=1, max_sentence_length=12000 |
| meta-llama/Llama-2-7b-chat-hf       | ragonize-Mistral-7B-Instruct-v0.1-test  | 0.494744 | -0.014927 | 0.592833                  | 0.438846                 | amount=-1, knowledge_layers=[0, 32], top_n_heads=1, top_n_layers=10, external_sim_scaling=0.2, param_know_scaling=1, max_sentence_length=12000 |
| meta-llama/Meta-Llama-3-8B-Instruct | ragonize-Mistral-7B-Instruct-v0.1-test  | 0.428484 | -0.087025 | 0.473242                  | 0.449086                 | amount=-1, knowledge_layers=[0, 32], top_n_heads=3, top_n_layers=30, external_sim_scaling=0.4, param_know_scaling=1, max_sentence_length=12000 |
| meta-llama/Llama-2-13b-chat-hf      | ragonize-Mistral-7B-Instruct-v0.1-train | 0.613052 | 0.180416  | 0.599783                  | 0.457712                 | amount=-1, knowledge_layers=[0, 32], top_n_heads=2, top_n_layers=17, external_sim_scaling=0.6, param_know_scaling=1, max_sentence_length=12000 |
| meta-llama/Llama-2-7b-chat-hf       | ragonize-Mistral-7B-Instruct-v0.1-train | 0.466448 | -0.053953 | 0.580402                  | 0.441464                 | amount=-1, knowledge_layers=[0, 32], top_n_heads=1, top_n_layers=10, external_sim_scaling=0.2, param_know_scaling=1, max_sentence_length=12000 |
| meta-llama/Meta-Llama-3-8B-Instruct | ragonize-Mistral-7B-Instruct-v0.1-train | 0.428841 | -0.093502 | 0.473280                  | 0.451548                 | amount=-1, knowledge_layers=[0, 32], top_n_heads=3, top_n_layers=30, external_sim_scaling=0.4, param_know_scaling=1, max_sentence_length=12000 |
| meta-llama/Llama-2-13b-chat-hf      | ragonize-Mistral-7B-Instruct-v0.3-test  | 0.610577 | 0.170430  | 0.608724                  | 0.450505                 | amount=-1, knowledge_layers=[0, 32], top_n_heads=2, top_n_layers=17, external_sim_scaling=0.6, param_know_scaling=1, max_sentence_length=12000 |
| meta-llama/Llama-2-7b-chat-hf       | ragonize-Mistral-7B-Instruct-v0.3-test  | 0.549516 | 0.059588  | 0.592308                  | 0.445506                 | amount=-1, knowledge_layers=[0, 32], top_n_heads=1, top_n_layers=10, external_sim_scaling=0.2, param_know_scaling=1, max_sentence_length=12000 |
| meta-llama/Meta-Llama-3-8B-Instruct | ragonize-Mistral-7B-Instruct-v0.3-test  | 0.432055 | -0.079867 | 0.474916                  | 0.450368                 | amount=-1, knowledge_layers=[0, 32], top_n_heads=3, top_n_layers=30, external_sim_scaling=0.4, param_know_scaling=1, max_sentence_length=12000 |
| meta-llama/Llama-2-13b-chat-hf      | ragonize-Mistral-7B-Instruct-v0.3-train | 0.600595 | 0.155873  | 0.597375                  | 0.457304                 | amount=-1, knowledge_layers=[0, 32], top_n_heads=2, top_n_layers=17, external_sim_scaling=0.6, param_know_scaling=1, max_sentence_length=12000 |
| meta-llama/Llama-2-7b-chat-hf       | ragonize-Mistral-7B-Instruct-v0.3-train | 0.487770 | -0.020886 | 0.579786                  | 0.444378                 | amount=-1, knowledge_layers=[0, 32], top_n_heads=1, top_n_layers=10, external_sim_scaling=0.2, param_know_scaling=1, max_sentence_length=12000 |
| meta-llama/Meta-Llama-3-8B-Instruct | ragonize-Mistral-7B-Instruct-v0.3-train | 0.435583 | -0.079903 | 0.474978                  | 0.454994                 | amount=-1, knowledge_layers=[0, 32], top_n_heads=3, top_n_layers=30, external_sim_scaling=0.4, param_know_scaling=1, max_sentence_length=12000 |
| meta-llama/Llama-2-13b-chat-hf      | ragtruth-llama-2-7b-chat-test           | 0.655815 | 0.242899  | 0.575256                  | 0.497248                 | amount=-1, knowledge_layers=[0, 32], top_n_heads=2, top_n_layers=17, external_sim_scaling=0.6, param_know_scaling=1, max_sentence_length=12000 |
| meta-llama/Llama-2-7b-chat-hf       | ragtruth-llama-2-7b-chat-test           | 0.462686 | -0.041230 | 0.554141                  | 0.470774                 | amount=-1, knowledge_layers=[0, 32], top_n_heads=1, top_n_layers=10, external_sim_scaling=0.2, param_know_scaling=1, max_sentence_length=12000 |
| meta-llama/Llama-2-7b-chat-hf       | ragtruth-llama-2-7b-chat-train          | 0.470548 | -0.047593 | 0.541236                  | 0.483425                 | amount=-1, knowledge_layers=[0, 32], top_n_heads=1, top_n_layers=10, external_sim_scaling=0.2, param_know_scaling=1, max_sentence_length=12000 |
| meta-llama/Llama-2-7b-chat-hf       | ragtruth-mistral-7B-instruct-test       | 0.386935 | -0.213502 | 0.547003                  | 0.469292                 | amount=-1, knowledge_layers=[0, 32], top_n_heads=1, top_n_layers=10, external_sim_scaling=0.2, param_know_scaling=1, max_sentence_length=12000 |
| meta-llama/Llama-2-7b-chat-hf       | ragtruth-mistral-7B-instruct-train      | 0.447357 | -0.119405 | 0.545136                  | 0.477901                 | amount=-1, knowledge_layers=[0, 32], top_n_heads=1, top_n_layers=10, external_sim_scaling=0.2, param_know_scaling=1, max_sentence_length=12000 |
| meta-llama/Llama-2-13b-chat-hf      | test_llama2-13b                         | 0.649758 | 0.192644  | 0.613209                  | 0.431378                 | amount=-1, knowledge_layers=[0, 32], top_n_heads=2, top_n_layers=17, external_sim_scaling=0.6, param_know_scaling=1, max_sentence_length=12000 |
| meta-llama/Llama-2-7b-chat-hf       | test_llama2-13b                         | 0.223430 | -0.331894 | 0.598724                  | 0.365325                 | amount=-1, knowledge_layers=[0, 32], top_n_heads=1, top_n_layers=10, external_sim_scaling=0.2, param_know_scaling=1, max_sentence_length=12000 |
| meta-llama/Meta-Llama-3-8B-Instruct | test_llama2-13b                         | 0.361111 | -0.165995 | 0.489833                  | 0.430050                 | amount=-1, knowledge_layers=[0, 32], top_n_heads=3, top_n_layers=30, external_sim_scaling=0.4, param_know_scaling=1, max_sentence_length=12000 |
| meta-llama/Llama-2-13b-chat-hf      | test_llama2-7b                          | 0.595763 | 0.168798  | 0.588993                  | 0.477992                 | amount=-1, knowledge_layers=[0, 32], top_n_heads=2, top_n_layers=17, external_sim_scaling=0.6, param_know_scaling=1, max_sentence_length=12000 |
| meta-llama/Llama-2-7b-chat-hf       | test_llama2-7b                          | 0.380508 | -0.187129 | 0.581691                  | 0.411088                 | amount=-1, knowledge_layers=[0, 32], top_n_heads=1, top_n_layers=10, external_sim_scaling=0.2, param_know_scaling=1, max_sentence_length=12000 |
| meta-llama/Meta-Llama-3-8B-Instruct | test_llama2-7b                          | 0.451695 | -0.066342 | 0.506387                  | 0.448096                 | amount=-1, knowledge_layers=[0, 32], top_n_heads=3, top_n_layers=30, external_sim_scaling=0.4, param_know_scaling=1, max_sentence_length=12000 |
| meta-llama/Llama-2-13b-chat-hf      | test_mistral-7b                         | 0.621739 | 0.141331  | 0.619123                  | 0.452662                 | amount=-1, knowledge_layers=[0, 32], top_n_heads=2, top_n_layers=17, external_sim_scaling=0.6, param_know_scaling=1, max_sentence_length=12000 |
| meta-llama/Llama-2-7b-chat-hf       | test_mistral-7b                         | 0.242029 | -0.307398 | 0.548017                  | 0.376977                 | amount=-1, knowledge_layers=[0, 32], top_n_heads=1, top_n_layers=10, external_sim_scaling=0.2, param_know_scaling=1, max_sentence_length=12000 |
| meta-llama/Meta-Llama-3-8B-Instruct | test_mistral-7b                         | 0.411594 | -0.069986 | 0.491207                  | 0.466147                 | amount=-1, knowledge_layers=[0, 32], top_n_heads=3, top_n_layers=30, external_sim_scaling=0.4, param_know_scaling=1, max_sentence_length=12000 |
| meta-llama/Llama-2-13b-chat-hf      | val_llama2-13b                          | 0.681667 | 0.275984  | 0.587174                  | 0.439216                 | amount=-1, knowledge_layers=[0, 32], top_n_heads=2, top_n_layers=17, external_sim_scaling=0.6, param_know_scaling=1, max_sentence_length=12000 |
| meta-llama/Llama-2-7b-chat-hf       | val_llama2-13b                          | 0.324167 | -0.262293 | 0.554994                  | 0.379295                 | amount=-1, knowledge_layers=[0, 32], top_n_heads=1, top_n_layers=10, external_sim_scaling=0.2, param_know_scaling=1, max_sentence_length=12000 |
| meta-llama/Meta-Llama-3-8B-Instruct | val_llama2-13b                          | 0.350000 | -0.223216 | 0.472528                  | 0.426269                 | amount=-1, knowledge_layers=[0, 32], top_n_heads=3, top_n_layers=30, external_sim_scaling=0.4, param_know_scaling=1, max_sentence_length=12000 |
| meta-llama/Llama-2-13b-chat-hf      | val_llama2-7b                           | 0.567460 | 0.125494  | 0.534170                  | 0.480801                 | amount=-1, knowledge_layers=[0, 32], top_n_heads=2, top_n_layers=17, external_sim_scaling=0.6, param_know_scaling=1, max_sentence_length=12000 |
| meta-llama/Llama-2-7b-chat-hf       | val_llama2-7b                           | 0.404762 | -0.143317 | 0.530348                  | 0.450229                 | amount=-1, knowledge_layers=[0, 32], top_n_heads=1, top_n_layers=10, external_sim_scaling=0.2, param_know_scaling=1, max_sentence_length=12000 |
| meta-llama/Meta-Llama-3-8B-Instruct | val_llama2-7b                           | 0.580357 | 0.126987  | 0.510134                  | 0.528875                 | amount=-1, knowledge_layers=[0, 32], top_n_heads=3, top_n_layers=30, external_sim_scaling=0.4, param_know_scaling=1, max_sentence_length=12000 |
| meta-llama/Llama-2-13b-chat-hf      | val_mistral-7b                          | 0.373016 | -0.122110 | 0.534800                  | 0.442726                 | amount=-1, knowledge_layers=[0, 32], top_n_heads=2, top_n_layers=17, external_sim_scaling=0.6, param_know_scaling=1, max_sentence_length=12000 |
| meta-llama/Llama-2-7b-chat-hf       | val_mistral-7b                          | 0.403175 | -0.098447 | 0.562352                  | 0.429133                 | amount=-1, knowledge_layers=[0, 32], top_n_heads=1, top_n_layers=10, external_sim_scaling=0.2, param_know_scaling=1, max_sentence_length=12000 |
| meta-llama/Meta-Llama-3-8B-Instruct | val_mistral-7b                          | 0.357143 | -0.140725 | 0.469622                  | 0.437029                 | amount=-1, knowledge_layers=[0, 32], top_n_heads=3, top_n_layers=30, external_sim_scaling=0.4, param_know_scaling=1, max_sentence_length=12000 |