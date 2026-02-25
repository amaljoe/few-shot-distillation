## Compute Setup

- Node: `cn14-dgx` — 4×A100 80GB
- SSH: `ssh -p 4422 cn14-dgx` (port 4422, not 22)
- Local and compute node share the same filesystem; anything that doesn't need GPU can be run locally

## tmux Sessions

Start each session: `app` alias → apptainer → `conda activate /dev/shm/vllm` → `cd ~/workspace/icl-distillation`

| Session | Purpose |
|---------|---------|
| `claude` | Primary session — training, precompute, eval scripts |
| `vllm` | vLLM server |
| `vscode` | Parallel training (second condition) |
| `tensor` | TensorBoard |

Create additional sessions as needed following the same setup pattern.

## GPU Allocation Convention

| GPUs | Use |
|------|-----|
| 0, 1 | Condition B training / parallel training |
| 2, 3 | Condition C training / parallel training |
| 0–3 | vLLM server (tensor-parallel-size 4) or single training |

Check GPU status: `nvidia-smi --query-gpu=index,memory.used --format=csv,noheader`

## vLLM Notes

- **vLLM binds to the container's localhost.** API calls from an SSH shell (outside the
  container) time out for POST requests even though GET (e.g. `/health`, `/v1/models`) works.
  Always call `scripts/evaluate.py` and curl from inside a tmux session (apptainer env).
- **First LoRA request is slow** — vLLM compiles CUDA kernels for the adapter on the first
  call. Wait ≥60 s before concluding a hang. Subsequent requests are fast.
- **tensor-parallel-size 4 recommended** for serving on this node (80 GB × 4). Use
  `--max-model-len 2048` for zero-shot eval; `4096` for 8-shot prompts (~2077 tokens).
- **Port conflicts**: accelerate may warn about port 29500 in use — non-fatal, it finds
  another port automatically. Use `--main_process_port 29501/29502/...` to separate
  parallel training runs.

## 8-shot Prompt Lengths

GSM8K 8-shot prompts are ~2077 tokens with Qwen3-1.7B tokenizer. Always set
`--max_model_len 4096` (not 2048) for any script that processes teacher sequences.

## Training Speed

- Qwen3-1.7B + LoRA (r=16), 2×A100, bf16: ~1.1–1.2 it/s
- 1000 steps ≈ 14–15 minutes per condition
- Effective batch size: 4 (per device) × 2 (GPUs) × 4 (grad accum) = 32 per step

## tmux Pane Tips

- If `tmux capture-pane` shows old scrollback content and the pane appears frozen, the pane
  may be in copy/scroll mode. Send `q` first to exit: `tmux send-keys -t SESSION q`
- Always use `SESSION:window.pane` notation (e.g. `claude:0.0`) for reliability
- After a long-running script finishes, send a blank Enter to confirm the prompt is active
  before sending the next command

## Qwen3 Model Notes

- Qwen3-1.7B: 28 layers, hidden_size=2048 (not 1536 as listed in some docs)
- Always disable thinking mode: `enable_thinking=False` in `apply_chat_template`, or
  `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` in vLLM API
- Thinking tokens (`<think>…</think>`) break `#### <number>` answer extraction

## Accelerate Config

Written to `~/.cache/huggingface/accelerate/default_config.yaml` for 2-GPU bf16 setup.
When running two accelerate jobs in parallel, pass different `--main_process_port` values
to avoid port collision.

## Other Notes

- `sleep less` — avoid polling loops; use background tasks with notification instead
- Other users may be present on the node (`pred2` session observed). Check
  `nvidia-smi` before assuming GPUs are free.
- The `datasets`, `omegaconf`, `peft`, `accelerate`, `tensorboard` packages may need to be
  installed in the mamba env if not present: `pip install <pkg> -q`
