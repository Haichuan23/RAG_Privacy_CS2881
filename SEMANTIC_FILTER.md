# Semantic Filter Usage Guide

## Usage

1. Edit `semantic_guard_main.sh` and update the following variables to run new models.

For example, if you want to run Qwen/Qwen2.5-7B-Instruct model via Together API.
```bash
API=together
HF_MODEL=Qwen/Qwen2.5-7B-Instruct
TOGETHER_MODEL=Qwen/Qwen2.5-7B-Instruct-Turbo
```

2. Change input file name and datastore root according to whether you want to replicate
the original paper's result, or you want to test on robustness or benign input.

```bash
IO_INPUT_PATH="anchor_prompts_benign.json"
DATASTORE_ROOT="./benign"
```

Also, change the evaluation mode of the semantic main file correctly:
```bash
--evaluation_mode benign
```

3. Running the scripts on Harvard FASRC cluster
```bash
sbatch semantic_guard_main.sh
```