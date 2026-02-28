# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

All GPU work runs on `cn14-dgx` (4×A100 80GB). Connect via `ssh -p 4422 cn14-dgx`.
The local and compute nodes share the same filesystem — scripts that don't need GPU can run locally.

On cn14-dgx, sessions start via the `app` alias → apptainer container → `mamba activate /dev/shm/vllm`. All Python commands below assume this environment is active.

Primary tmux sessions: `claude` (training/eval), `vscode` (parallel training), `vllm` (vLLM server), `tensor` (TensorBoard).

Check GPU availability before launching jobs:
```bash
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
```
Other users may share the node; verify GPUs are free before allocating them.

## Common Commands

### Evaluate ICL gap (before training)
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_icl.py --model Qwen/Qwen3-1.7B --n_samples 1319
```

### Train SFT baseline (2 GPUs)
```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --num_processes 2 --mixed_precision bf16 --main_process_port 29500 \
    src/training/train_baseline.py --config configs/qwen1b7.yaml \
    --output_dir experiments/qwen1b7/baseline
```

### Train with distillation (2 GPUs, can run in parallel with baseline on GPUs 2,3)
```bash
CUDA_VISIBLE_DEVICES=2,3 accelerate launch \
    --num_processes 2 --mixed_precision bf16 --main_process_port 29501 \
    src/training/train_online_v1.py \
    --base_config configs/qwen1b7.yaml --config configs/online_v1.yaml \
    --output_dir experiments/qwen1b7/online_v1
```
Note: `train_online_v1.py` takes **two** config flags (`--base_config` + `--config`), merged via OmegaConf. `train_baseline.py` takes only `--config`.

### Evaluate checkpoints
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/eval_checkpoints.py \
    --config configs/qwen1b7.yaml \
    --conditions baseline online_v1 \
    --base_dir experiments/qwen1b7 \
    --n_samples 1319 --checkpoint_steps 200 400 600 800 1000 \
    --output experiments/qwen1b7/results.json \
    --tensor_parallel_size 4
```

### Generate figures (no GPU needed, run locally or via claude session)
```bash
python scripts/gen_summary_fig.py      # cross-model bar chart → assets/summary_comparison.png
python scripts/gen_main_fig.py         # Qwen3-1.7B checkpoint curve
python scripts/gen_ablation_fig.py     # ablation curve + bar chart
python scripts/gen_loss_fig.py         # CE loss comparison (reads TensorBoard logs)
python scripts/write_results.py        # write results.md from JSON eval outputs
```

### Evaluate checkpoints (HF fallback — for non-vLLM models like Param2-17B)
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/eval_hf_checkpoints.py \
    --config configs/param2_17b.yaml \
    --conditions baseline online_v1 \
    --base_dir experiments/param2_17b \
    --n_samples 1319 --checkpoint_steps 200 400 600 800 1000 \
    --output experiments/param2_17b_eval.json
```
Use `eval_hf_icl.py` (analogous to `eval_icl.py`) for the ICL baseline of non-vLLM models.

### Causal ablation training
```bash
# Uses train_ablation.py; --condition_name controls which ablation
CUDA_VISIBLE_DEVICES=0,1 accelerate launch ... src/training/train_ablation.py \
    --condition_name zeroshot_teacher \
    --base_config configs/qwen1b7.yaml --config configs/ablation_zeroshot_teacher.yaml \
    --output_dir experiments/ablations
```

### Full experiment sequences (orchestration scripts)
```bash
bash scripts/train_qwen8b.sh   # Qwen3-8B: baseline + distill in parallel, then ablation + eval
bash scripts/eval_ablations.sh # Evaluate Qwen3-1.7B SFT + ablation conditions
```

## Architecture

### Config system
Configs are YAML files loaded with OmegaConf. Two-level inheritance:
- **Base config** (`configs/{model}.yaml`): model arch (`name`, `num_layers`, `hidden_size`), data settings, training hyperparams, LoRA config.
- **Override config** (`configs/online_v1_{model}.yaml`): distillation-specific fields (`distillation.n_top_logits`, `distillation.lambda_distill`) and output_dir override.

`train_online_v1.py` merges them: `cfg = OmegaConf.merge(base_cfg, v1_cfg)`. `train_baseline.py` uses only the base config.

Full fine-tuning is toggled via `training.use_lora: false` in a config (Gemma-3-270M). All three training scripts and `eval_checkpoints.py` support both modes.

### Training scripts

**`src/training/train_baseline.py`** — CE-only SFT. Loads `StudentModel`, runs standard HuggingFace forward pass with labels, cosine LR schedule. Output path: `{output_dir}/baseline/checkpoint-{step}`.

**`src/training/train_online_v1.py`** — Few-shot distillation. Loads a **frozen teacher** (same base model, no LoRA, one copy per DDP process) alongside the LoRA student. At each step: teacher forward (no_grad, 8-shot context) → top-K teacher logits → student forward (CE loss) → MSE on top-K positions → `L = L_CE + λ·MSE`. Output path: `{output_dir}/online_v1/checkpoint-{step}`.

**`src/training/train_ablation.py`** — Same as V1 but supports three teacher modes (`zeroshot_teacher`, `shuffled_answers`, `fewshot_teacher`). Output path: `{output_dir}/{condition_name}/{condition_name}/checkpoint-{step}`.

### Token alignment (critical)
Teacher sequence: `[8-shot context] + [question] + [answer_tokens]`
Student sequence: `[question] + [answer_tokens]`

Both end with the **same answer token IDs**. `answer_alignment()` in `train_online_v1.py` computes this as:
- `n_ans = (labels != -100).sum(dim=1)` — number of answer tokens per example
- `t_ans_start = t_lens - n_ans` — teacher's first answer token position
- `s_ans_start = (labels != -100).float().argmax(dim=1)` — student's first answer token

The dataset uses `teacher_include_answer=True` for online training (the teacher sequence includes the gold answer for suffix alignment).

### Data pipeline (`src/data/gsm8k_loader.py`)
`GSM8KDistillDataset` returns paired teacher/student inputs per example. Few-shot examples are sampled deterministically (`random.Random(seed + idx)`). `collate_fn` pads within a batch. Key dataset flag: `shuffle_fewshot_answers=True` activates the shuffled-answer ablation condition.

### Checkpoint structure
`eval_checkpoints.py` expects: `{base_dir}/{condition_name}/{condition_name}/checkpoint-{step}`

This means `--output_dir experiments/qwen1b7/baseline` → scripts write to `experiments/qwen1b7/baseline/baseline/checkpoint-{step}`. The condition name is appended once by the training script, resulting in the doubled directory.

Full-FT vs LoRA auto-detection: presence of `adapter_config.json` in the checkpoint directory.

### Model specs
| Model | Layers | Hidden size | Note |
|---|---|---|---|
| Qwen3-1.7B | 28 | 2048 | hidden_size is 2048, not 1536 |
| Qwen3-8B | 36 | 4096 | |
| Llama-3.2-3B | — | — | |
| Gemma-3-270M | — | — | full FT (no LoRA) |

### Experiment naming
| Model | Base config | Online config | Experiment dir |
|---|---|---|---|
| Qwen3-1.7B | `configs/qwen1b7.yaml` | `configs/online_v1.yaml` | `experiments/qwen1b7/` → symlink to `experiments/poc/` |
| Qwen3-8B | `configs/qwen8b.yaml` | `configs/online_v1_qwen8b.yaml` | `experiments/qwen8b/` |
| Llama-3.2-3B | `configs/llama3b.yaml` | `configs/online_v1_llama.yaml` | `experiments/llama3b/` |
| Gemma-3-270M | `configs/gemma270m.yaml` | `configs/gemma270m_distill.yaml` | `experiments/gemma270m/` |

`configs/base.yaml` is kept for backward compatibility with existing experiment outputs; new work uses `configs/qwen1b7.yaml`.

## Critical Implementation Details

**LoRA + gradient checkpointing**: always call `model.enable_input_require_grads()` **after** `model.gradient_checkpointing_enable()`. Without this, loss has no `grad_fn` and training crashes silently. Applied in all three training scripts.

**Qwen3 thinking mode**: always pass `enable_thinking=False` to `apply_chat_template` (or `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` in vLLM API). Thinking tokens break `#### <number>` answer extraction.

**vLLM context**: vLLM binds to the container's localhost. Always call `evaluate.py` and any vLLM API client from **inside** the apptainer session (not from a plain SSH shell). The first LoRA request compiles CUDA kernels — wait ≥60 s before concluding a hang; subsequent requests are fast.

**Parallel training ports**: use `--main_process_port 29500` on GPUs 0,1 and `--main_process_port 29501` on GPUs 2,3 to avoid collision when running two accelerate jobs simultaneously.

**GSM8K answer extraction**: `re.search(r"####\s*([\d,]+)", text)` — this pattern is used in eval scripts and must match the model's output format.

## Paper Compilation

The paper lives in `paper/` (gitignored).

### Compile PDF (must be inside the apptainer container)

`pdflatex` is only available inside the apptainer container (`app` alias). LaTeX packages are installed in `~/images/mine.def` (texlive-latex-base, texlive-latex-extra, texlive-fonts-extra, texlive-bibtex-extra, texlive-science, cm-super, etc.).

```bash
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```
