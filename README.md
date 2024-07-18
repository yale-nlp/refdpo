# RefDPO: Understanding Reference Policies in Direct Preference Optimization

## Quick Links

- [Installation](#installation)
- [Running the code](#running-the-code)
- [Datasets](#datasets)
- [Experimental Results](#experimental-results)
    - [RQ1: What Is the Optimal KL Constraint Strength for DPO?](#rq1-what-is-the-optimal-kl-constraint-strength-for-dpo)
    - [RQ2: Is a Reference Policy Necessary for Effective Preference Learning?](#rq2-is-a-reference-policy-necessary-for-effective-preference-learning)
    - [RQ3: Does DPO Benefit from Stronger Reference Policies?](#rq3-does-dpo-benefit-from-stronger-reference-policies)

## Installation

Our code base is based on Huggingface's Transformers library, deepspeed, and PyTorch.
To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
```

Our code base is adapted from the the [open-instruct](https://github.com/allenai/open-instruct) repository.

## Running the code

To run the code, you can use the following command (it assumes there are 8 GPUs available):

```bash
accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes 8 \
    --use_deepspeed \
    --deepspeed_config_file deepspeed.conf \
    main.py \
    --cuda \
    --dataset 'yale-nlp/RefDPO' \
    --data_split 'mistral' \
    --epoch 3 \
    --beta 0.1 \
    --dpo_weight 1.0 \
    --model_type 'HuggingFaceH4/mistral-7b-sft-beta' \
    --insert_eos \
    -l
```

Each argument is explained in the `main.py` file.

To run the training *without* the reference model, you can use the `--ref_free` flag:

```bash
accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes 8 \
    --use_deepspeed \
    --deepspeed_config_file deepspeed.conf \
    main.py \
    --cuda \
    --dataset 'yale-nlp/RefDPO' \
    --data_split 'mistral' \
    --epoch 3 \
    --beta 10.0 \
    --dpo_weight 1.0 \
    --ref_free \
    --model_type 'HuggingFaceH4/mistral-7b-sft-beta' \
    --insert_eos \
    -l
```

### Code structure

- `main.py`: The main file to run the code.
- `data_utils.py`: The data processing utilities.
- `utils.py`: Utility functions.
- `deepspeed.conf`: The deepspeed configuration file.
- `dpo_utils.py`: The DPO utilities.

### Resource requirements

Our training requires 8 GPUs with 48GB of memory each. We use the `deepspeed` library to distribute the training across multiple GPUs.


## Datasets

We have made the datasets used in the paper available on Huggingface's dataset hub: [yale-nlp/RefDPO](https://huggingface.co/datasets/yale-nlp/RefDPO).
It contains 5 different datasets.
Each dataset is built upon the [UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback) dataset, specifically its binarized version [ultrafeedback_binarized_cleaned](https://huggingface.co/datasets/allenai/ultrafeedback_binarized_cleaned) converted from [ultrafeedback_binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized).
The datasets contain **pre-computed log-probabilities** of the reference policy/model for the output pairs in the UltraFeedback dataset.

| Dataset | Reference Model | Description | 
|---------|-----------------|-------------|
| `mistral` | [HuggingFaceH4/mistral-7b-sft-beta](https://huggingface.co/HuggingFaceH4/mistral-7b-sft-beta) | The log-probabilities are computed using the Mistral-7B-SFT model. |
| `tulu2` | [allenai/tulu-2-7b](https://huggingface.co/allenai/tulu-2-7b) | The log-probabilities are computed using the Tulu-2-7B model. |
| `mistral_prior` | [HuggingFaceH4/mistral-7b-sft-beta](https://huggingface.co/HuggingFaceH4/mistral-7b-sft-beta) | The **prior** (unconditional) log-probabilities are computed using the Mistral-7B-SFT model. |
| `mistralv2` | [mistralai/Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) | The log-probabilities are computed using the Mistral-7B-Instruct-v0.2 model. |
| `llama3` | [meta-llama/Meta-Llama-3-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct) | The log-probabilities are computed using the Meta-Llama-3-70B-Instruct model. |


## Experimental Results

Below are the model checkpoints for the models trained in the paper.


### RQ1: What Is the Optimal KL Constraint Strength for DPO?

Below are the model checkpoints fine-tuned with DPO from [mistral-7b-sft](https://huggingface.co/HuggingFaceH4/mistral-7b-sft-beta) and [tulu-2-7b](https://huggingface.co/allenai/tulu-2-7b). The models are fine-tuned with different KL constraint strengths (**$\beta$**). 

The checkpoints are available on Huggingface's model hub. They are evaluated using the length-controlled AlpacaEval2 score [reference](https://arxiv.org/abs/2404.04475).


#### Checkpoints fine-tuned from [mistral-7b-sft](https://huggingface.co/HuggingFaceH4/mistral-7b-sft-beta)


| $\beta$  | HF Checkpoint | AlpacaEval2 LC-Score |
|-------|---------------|-----------------|
| 0.1   | [yale-nlp/mistral-7b-dpo-beta-0.1](https://huggingface.co/yale-nlp/mistral-7b-dpo-beta-0.1)         | 14.03           |
| 0.05  | [yale-nlp/mistral-7b-dpo-beta-0.05](https://huggingface.co/yale-nlp/mistral-7b-dpo-beta-0.05)         | 13.29          |
| 0.02  | [yale-nlp/mistral-7b-dpo-beta-0.02](https://huggingface.co/yale-nlp/mistral-7b-dpo-beta-0.02)         | 16.06           |
| 0.01  | [yale-nlp/mistral-7b-dpo-beta-0.01](https://huggingface.co/yale-nlp/mistral-7b-dpo-beta-0.01)         | **16.25**         |
| 0.005 | [yale-nlp/mistral-7b-dpo-beta-0.005](https://huggingface.co/yale-nlp/mistral-7b-dpo-beta-0.005)         | 12.36           |


#### Checkpoints fine-tuned from [tulu-2-7b](https://huggingface.co/allenai/tulu-2-7b)

| $\beta$  | HF Checkpoint | AlpacaEval2 LC-Score |
|-------|---------------|-----------------|
| 0.1   |  [yale-nlp/tulu2-7b-dpo-beta-0.1](https://huggingface.co/yale-nlp/tulu2-7b-dpo-beta-0.1)           | 9.38 |
| 0.05  |   [WIP]           |  9.96 |
| 0.02  |  [yale-nlp/tulu2-7b-dpo-beta-0.02](https://huggingface.co/yale-nlp/tulu2-7b-dpo-beta-0.02)           | **10.46** |
| 0.01  | [WIP]           | 7.86 |
| 0.005 |  [yale-nlp/tulu2-7b-dpo-beta-0.005](https://huggingface.co/yale-nlp/tulu2-7b-dpo-beta-0.005)           | [degenerate] |


### RQ2: Is a Reference Policy Necessary for Effective Preference Learning?

Below are the optimal checkpoints fine-tuned from [mistral-7b-sft](https://huggingface.co/HuggingFaceH4/mistral-7b-sft-beta) with three different reward parameterizations.

| Reward Parameterization | HF Checkpoint | AlpacaEval2 LC-Score | $\beta$ |
|-------|---------------|-----------------| ---- |
| $\beta\frac{p_\theta(y\|x)}{p_{\mathrm{ref}}(y\|x)}$ (DPO) | [yale-nlp/mistral-7b-dpo-beta-0.01](https://huggingface.co/yale-nlp/mistral-7b-dpo-beta-0.01)   | 16.25 | 0.01 |
| $\beta p_\theta(y\|x)$ (Posterior Probability) | [yale-nlp/mistral-probability](https://huggingface.co/yale-nlp/mistral-probability)   | 12.84 | 100.0 |
| $\beta p_\theta(x\|y)$ (Likelihood Function) | [yale-nlp/mistral-likelihood](https://huggingface.co/yale-nlp/mistral-likelihood)   | 13.63 | 0.01 |

### RQ3: Does DPO Benefit from Stronger Reference Policies?

Here we use two stronger reference models [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) and [Meta-Llama-3-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct) for DPO training.


#### Checkpoints fine-tuned from [mistral-7b-sft](https://huggingface.co/HuggingFaceH4/mistral-7b-sft-beta) using [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) as the reference policy.


| $\beta$  | HF Checkpoint | AlpacaEval2 LC-Score |
|-------|---------------|-----------------|
| 10.0   | [yale-nlp/mistral-7b-dpo-mistralv2-7b-beta-10.0](https://huggingface.co/yale-nlp/mistral-7b-dpo-mistralv2-7b-beta-10.0)         | 18.74           |
| 1.00  | [yale-nlp/mistral-7b-dpo-mistralv2-7b-beta-1.0](https://huggingface.co/yale-nlp/mistral-7b-dpo-mistralv2-7b-beta-1.0)         | **20.25**          |
| 0.10  | [yale-nlp/mistral-7b-dpo-mistralv2-7b-beta-0.1](https://huggingface.co/yale-nlp/mistral-7b-dpo-mistralv2-7b-beta-0.1)         | 19.58           |
| 0.01  | [yale-nlp/mistral-7b-dpo-mistralv2-7b-beta-0.01](https://huggingface.co/yale-nlp/mistral-7b-dpo-mistralv2-7b-beta-0.01)         | 17.18         |
| 0.005 | [yale-nlp/mistral-7b-dpo-mistralv2-7b-beta-0.005](https://huggingface.co/yale-nlp/mistral-7b-dpo-mistralv2-7b-beta-0.005)         | 15.34           |


#### Checkpoints fine-tuned from [mistral-7b-sft](https://huggingface.co/HuggingFaceH4/mistral-7b-sft-beta) using [Meta-Llama-3-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct) as the reference policy.


| $\beta$  | HF Checkpoint | AlpacaEval2 LC-Score |
|-------|---------------|-----------------|
| 10.0   | [yale-nlp/mistral-7b-dpo-llama3-70b-beta-10.0](https://huggingface.co/yale-nlp/mistral-7b-dpo-llama3-70b-beta-10.0)         | 13.29           |
| 1.00  | [yale-nlp/mistral-7b-dpo-llama3-70b-beta-1.0](https://huggingface.co/yale-nlp/mistral-7b-dpo-llama3-70b-beta-1.0)         | 9.59          |
| 0.10  | [yale-nlp/mistral-7b-dpo-llama3-70b-beta-0.1](https://huggingface.co/yale-nlp/mistral-7b-dpo-llama3-70b-beta-0.1)         | 10.99           |
| 0.01  | [yale-nlp/mistral-7b-dpo-llama3-70b-beta-0.01](https://huggingface.co/yale-nlp/mistral-7b-dpo-llama3-70b-beta-0.01)         | **15.37**         |
| 0.005 | [yale-nlp/mistral-7b-dpo-llama3-70b-beta-0.005](https://huggingface.co/yale-nlp/mistral-7b-dpo-llama3-70b-beta-0.005)         | 11.70          |


#### Checkpoints fine-tuned from [tulu-2-7b](https://huggingface.co/allenai/tulu-2-7b) using [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) as the reference policy.


| $\beta$  | HF Checkpoint | AlpacaEval2 LC-Score |
|-------|---------------|-----------------|
| 10.0   | [yale-nlp/tulu2-7b-dpo-mistralv2-7b-beta-10.0](https://huggingface.co/yale-nlp/tulu2-7b-dpo-mistralv2-7b-beta-10.0)         | 7.61          |
| 1.00  | [yale-nlp/tulu2-7b-dpo-mistralv2-7b-beta-1.0](https://huggingface.co/yale-nlp/tulu2-7b-dpo-mistralv2-7b-beta-1.0)         | **7.85**          |
| 0.10  | [yale-nlp/tulu2-7b-dpo-mistralv2-7b-beta-0.1](https://huggingface.co/yale-nlp/tulu2-7b-dpo-mistralv2-7b-beta-0.1)         | [degenerate]           |
| 0.01  | [yale-nlp/tulu2-7b-dpo-mistralv2-7b-beta-0.01](https://huggingface.co/yale-nlp/tulu2-7b-dpo-mistralv2-7b-beta-0.01)         | [degenerate]         |
| 0.005 | [WIP]         | [degenerate]           |


#### Checkpoints fine-tuned from [tulu-2-7b](https://huggingface.co/allenai/tulu-2-7b) using [Meta-Llama-3-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct) as the reference policy.


| $\beta$  | HF Checkpoint | AlpacaEval2 LC-Score |
|-------|---------------|-----------------|
| 10.0   | [yale-nlp/tulu2-7b-dpo-llama3-70b-beta-10.0](https://huggingface.co/yale-nlp/tulu2-7b-dpo-llama3-70b-beta-10.0)         | 9.79           |
| 1.00  | [yale-nlp/tulu2-7b-dpo-llama3-70b-beta-1.0](https://huggingface.co/yale-nlp/tulu2-7b-dpo-llama3-70b-beta-1.0)         | **11.17**          |
| 0.10  | [yale-nlp/tulu2-7b-dpo-llama3-70b-beta-0.1](https://huggingface.co/yale-nlp/tulu2-7b-dpo-llama3-70b-beta-0.1)         | 10.31           |
| 0.01  | [yale-nlp/tulu2-7b-dpo-llama3-70b-beta-0.01](https://huggingface.co/yale-nlp/tulu2-7b-dpo-llama3-70b-beta-0.01)         | 9.16         |
| 0.005 | [yale-nlp/tulu2-7b-dpo-llama3-70b-beta-0.005](https://huggingface.co/yale-nlp/tulu2-7b-dpo-llama3-70b-beta-0.005)         | 3.29          |


