Soft Prompting unlearning (SPUL) — Nevergrad (NES) instructions

This document explains how to run the ES (Nevergrad) variant of SPUL in this repository. The code implements a Nevergrad-based evolutionary search to optimize the soft prompt (prompt encoder / virtual tokens) instead of using backpropagation. This README focuses only on the ES/NES path (the rest of the repo still supports the original backprop Trainer).

Contents
- Overview
- Prerequisites
- Install
- Quick smoke test (small, fast)
- Full experiment (replicating paper-style evaluation)
- What to monitor during ES optimization
- How to evaluate forgetting and retention
- Reproducibility and tips
- Troubleshooting
- Saving and loading prompts

Overview
--------
SPUL originally uses a loss that combines three parts:
- fgt_ce_loss: cross-entropy loss on the forget set (samples in train_forget are assigned randomized labels)
- rtn_ce_loss: cross-entropy on the retain set (train_retain)
- rtn_kl_loss: KL between model logits with the prompt and the original model logits (optional regularizer)

For the ES variant we treat the prompt-encoder embeddings as a flat parameter vector and optimize these with Nevergrad (ask/tell style). The fitness for each candidate prompt is the same scalar SPUL loss (fgt_ce + alpha * rtn_ce + beta * rtn_kl), computed with forward-only passes (no gradients). Nevergrad minimizes that fitness.

Prerequisites
-------------
- Python 3.10 (the repo was tested on 3.10.x)
- GPU available and CUDA configured (the ES loop evaluates many forward passes; GPU is highly recommended)
- Hugging Face access if using models that require authentication

Main packages (see `requirements.txt`):
- torch
- transformers
- peft
- trl
- datasets
- evaluate
- nevergrad
- bitsandbytes (if using quantized models)
- wandb (optional for logging)

Install
-------
1. Create a virtualenv and activate it (zsh):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. If you already use a GPU environment with CUDA, ensure `torch` and `bitsandbytes` match your CUDA version.

Quick smoke test (very small, fast)
---------------------------------
Use a tiny ES run to confirm the end-to-end flow on a machine with GPU.

Example (very small budget):

```bash
python spul.py \
  --dataset sst2 \
  --model_checkpoints <ptuning_checkpoint_or_peft_dir> \
  --optimizer nevergrad \
  --es_pop_size 8 \
  --es_iters 10 \
  --es_eval_batches 2 \
  --es_batch_size 4 \
  --es_normalize \
  --ptuning_num_tokens 10 \
  --ptuning_hidden_size 64 \
  --lr 1e-4 \
  --train_batch_size 8 \
  --eval_batch_size 8 \
  --num_epochs 1 \
  --alpha 0.5 --beta 0.5 \
  --output_path es_test_out
```

Notes:
- `--model_checkpoints` should point to a PEFT directory produced by QLoRA (the repo's `qlora.py`) or a base model checkpoint combined with a ptuning checkpoint. The `get_ptuning_model` function expects a LoRA/PEFT base to wrap with PromptEncoderConfig.
- Start with small population and iterations for validation. ES is computationally expensive.

Full experiment (replicating paper evaluation)
---------------------------------------------
To mimic experiments reported in the paper, follow these steps:

1. Fine-tune base model with QLoRA (as in the README):
   - Use `qlora.py` to produce a fine-tuned PEFT checkpoint: keep batch sizes, epochs, and LoRA settings similar to the paper.
2. Prepare original logits:
   - SPUL's KL regularizer requires saved original logits from the base model (without prompt). The `spul.py` code will compute and save logits in `saved_logits/` if not present. You can precompute them using `get_logits_from_base_model` helper by running the SPUL script once (it will save them automatically).
3. Run ES-based SPUL (main run):

```bash
python spul.py \
  --dataset sst2 \
  --model_checkpoints <qlora_checkpoint_dir> \
  --optimizer nevergrad \
  --es_pop_size 64 \
  --es_iters 500 \
  --es_eval_batches 8 \
  --es_batch_size 16 \
  --es_normalize \
  --es_opt CMA \
  --ptuning_num_tokens 30 \
  --ptuning_hidden_size 128 \
  --alpha 0.5 --beta 0.5 \
  --output_path unlearn_checkpoints/spul_es_experiment
```

- Tuning `es_pop_size` and `es_iters` is the main cost/quality trade-off.
- Use `es_eval_batches` big enough to provide a stable fitness estimate (but not so big that each candidate is prohibitively expensive).

What to monitor during ES optimization
-------------------------------------
1. Best fitness over time (printed every few iterations). This should decrease.
2. Training (per-candidate) loss components (if you want more detail, you can modify the code to log fgt_ce, rtn_ce, rtn_kl separately for the best candidate per iteration).
3. If using `wandb`, configure `WANDB_PROJECT` to log ES progress (you can log best_loss, average loss across population each iteration, etc.).

How to check if training is going right and verify results
---------------------------------------------------------
After ES finishes (or periodically during optimization), verify using the same evaluation scripts used by the paper.

1. Load the ES-optimized prompt and run inference using `inference_spul.py` (it expects the prompt saved as a PEFT checkpoint). Example:

```bash
python inference_spul.py --dataset sst2 --model_checkpoints <path_to_es_checkpoint_dir>
```

This script will evaluate `train_retain`, `train_forget`, `test_retain`, and `test_forget` splits and log metrics.

2. Metrics to check (paper-relevant):
- Forget accuracy / F1 on `train_forget` and `test_forget` (should drop compared to base model: lower accuracy means more forgetting). If you used random labels for forget set, higher loss / lower accuracy vs base indicates successful forgetting.
- Retain accuracy / F1 on `train_retain` and `test_retain` (should remain high; we want minimal degradation).
- KL divergence on retain set (if beta>0) — smaller KL indicates output distribution near original logits.

3. Compute an explicit forgetting score (example):
- Baseline: measure accuracy of base QLoRA model on the forget set (should be relatively high if memorized).
- ES result: measure accuracy of model with optimized prompt on forget set. The delta (drop in accuracy) is the forgetting amount.
- Utility preservation: measure the change in performance on retain/test_retain.

4. Reproduce paper plots:
- Make plots of trade-off curves (forgetting vs retention) by running ES with different values of alpha/beta or varying forget set sizes.

Saving and loading prompts (notes)
----------------------------------
 - The ES code attempts to save a PEFT checkpoint using `model.save_pretrained(args.output_path)` after optimization. If PEFT save fails, the code falls back to saving a tensor `prompt_embedding.pt` inside the `args.output_path` directory.
 - `inference_spul.py` now includes automatic loading of `prompt_embedding.pt` if present. It will try to match the saved tensor to a model parameter whose name contains `prompt`, `virtual`, or `prompt_encoder`, and also falls back to matching by the configured `ptuning_num_tokens`. If a match is found the tensor is copied into the model before evaluation. You do not need to modify inference code for this common case.

Reproducibility and tips
------------------------
- Set `--es_seed` and make sure to set seeds for numpy/torch. The ES loop in `spul.py` seeds numpy, random, and torch.
- Use `--es_normalize` to stabilize noisy fitness signals when using small batch evaluations.
- Start with small budgets locally to validate correctness, then scale up for reproduction-quality results.

Troubleshooting
---------------
- "nevergrad not found": ensure you installed requirements (`pip install nevergrad`).
- "Model checkpoint not found": confirm `--model_checkpoints` points to a PEFT directory or to a pretrained model path that the code expects.
- Very slow optimization: lower `es_pop_size` and `es_eval_batches`, increase `es_iters` gradually.
- Diverging/NaN fitness: try enabling `--es_normalize` or reduce learning budget/pop size.

Next steps / improvements
------------------------
- Parallelize candidate evaluations across multiple GPUs or processes to speed up ES.
- Implement XNES exact variant using Evosax if you need the exact algorithm (requires JAX). We currently use Nevergrad's CMA/OnePlusOne as practical ES solvers.
- Add detailed per-iteration logging of the loss components (fgt_ce, rtn_ce, rtn_kl) to better monitor optimization behavior.



