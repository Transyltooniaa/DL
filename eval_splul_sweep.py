# eval_spul_sweep.py

import os
import re
import csv
import torch
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import logging as hf_logging
from peft import PeftModel

# -----------------------------------------------------------------------------
# Minimal noise + allow legacy .pt loads inside transformers
# -----------------------------------------------------------------------------
hf_logging.set_verbosity_error()
os.environ["TRANSFORMERS_NO_SAFE_TORCH_LOAD"] = "1"

# -----------------------------------------------------------------------------
# CONFIG (edit paths if needed)
# -----------------------------------------------------------------------------
BASE_MODEL = "facebook/opt-1.3b"

QLORA_DIR = os.path.expanduser(
    "~/DeepLearning/Abhinav/qlora_checkpoints/opt-1.3b-qlora-sst2/final_model"
)

SPUL_DIR = "./unlearn_checkpoints/spul_opt1.3b_sst2_nes_fix7"   # directory with adapter_model.safetensors + prompt_checkpoint_iter_*.pt

DATASET_NAME = "karuna-bhaila/Unlearning_SST2v3"
MAX_LENGTH = 384
BATCH_SIZE = 8
MAX_NEW_TOKENS = 3

# Which checkpoints to evaluate.
# - If CHECKPOINT_ITERS is None -> auto-detect all "prompt_checkpoint_iter_*.pt" in SPUL_DIR.
# - Else use this list (ints).
CHECKPOINT_ITERS = [1,4,5,8,16,24,32,40,48,56,64,72,80]

# For which single checkpoint to dump ~30 examples to a file (set to an int in list above)
EXAMPLE_CHECKPOINT = 80
EXAMPLE_COUNT = 30
EXAMPLE_LOG_PATH = f"sample_outputs_iter_{EXAMPLE_CHECKPOINT}.txt"

RESULTS_CSV = "spul_sweep_results.csv"


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def list_available_iters(spul_dir):
    pat = re.compile(r"prompt_checkpoint_iter_(\d+)\.pt$")
    iters = []
    for fn in os.listdir(spul_dir):
        m = pat.match(fn)
        if m:
            iters.append(int(m.group(1)))
    return sorted(iters)


def load_stack():
    print("🔹 Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # decoder-only models need left padding for generation

    print("🔹 Loading base model...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to("cuda")

    print("🔹 Merging QLoRA adapter...")
    base = PeftModel.from_pretrained(base, QLORA_DIR)
    base = base.merge_and_unload()

    print("🔹 Attaching SPUL adapter...")
    model = PeftModel.from_pretrained(base, SPUL_DIR, is_trainable=False).to("cuda")
    model.eval()

    print("\n✅ Model stack ready (Base + QLoRA merged + SPUL adapter)\n")
    return model, tok


def print_prompt_stats(model):
    print("Checking prompt embedding statistics:")
    found = False
    for n, p in model.named_parameters():
        if "prompt" in n or "virtual" in n:
            print(f"  {n} | mean={p.data.mean().item():.6f}, std={p.data.std().item():.6f}")
            found = True
    if not found:
        print("  (No prompt/virtual params found — unexpected for SPUL adapter)")
    print("  If means/stds are ~0.000000, the prompt weights might not be loaded.\n")


def build_eval_split(tokenizer, split):
    EVAL_TEMPLATE = (
        "### Text: {text}\n\n### Question: What is the sentiment of the given text?\n\n### Sentiment:"
    )
    raw = load_dataset(DATASET_NAME, split=split)

    def _prep(ex):
        prompt = EVAL_TEMPLATE.format(text=ex["text"]).strip()
        enc = tokenizer(prompt, truncation=True, padding=False, max_length=MAX_LENGTH)
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "label": ex["label"],
            "orig_text": ex["text"],
        }

    ds = raw.map(_prep, remove_columns=raw.column_names)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"],output_all_columns=True,)
    return ds


def collate_fn(batch, tokenizer):
    texts = [b.get("orig_text", "") for b in batch]
    labels = torch.tensor([int(b["label"]) for b in batch], dtype=torch.long)
    enc = tokenizer.pad(
        [{"input_ids": b["input_ids"], "attention_mask": b["attention_mask"]} for b in batch],
        return_tensors="pt",
        padding=True,
    )
    enc["labels"] = labels
    enc["texts"] = texts
    return enc


def load_spul_prompt_checkpoint(model, spul_dir, iter_num):
    """Copy a saved ES checkpoint tensor into the SPUL prompt embedding."""
    ckpt_path = os.path.join(spul_dir, f"prompt_checkpoint_iter_{iter_num}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    t = torch.load(ckpt_path, map_location="cpu")
    matched = False
    for name, p in model.named_parameters():
        if "prompt_encoder" in name and "embedding" in name:
            if tuple(p.shape) == tuple(t.shape):
                p.data.copy_(t.to(p.device))
                matched = True
                print(f"✓ Loaded prompt checkpoint into '{name}' from {ckpt_path}")
                break
    if not matched:
        raise RuntimeError(
            f"Could not match checkpoint shape {tuple(t.shape)} to any prompt parameter."
        )


def classify_from_suffix(generated_suffix):
    s = generated_suffix.strip().lower()
    if s.startswith("positive"):
        return 1
    if s.startswith("negative"):
        return 0
    if s.startswith("neutral"):
        return 2
    # fallback: search anywhere
    if "positive" in s:
        return 1
    if "negative" in s:
        return 0
    if "neutral" in s:
        return 2
    return 2  # default to neutral


def evaluate_split(model, tokenizer, ds, max_new_tokens=MAX_NEW_TOKENS, batch_size=BATCH_SIZE,
                   keep_examples=False, max_examples=EXAMPLE_COUNT):
    loader = DataLoader(ds, batch_size=batch_size, collate_fn=lambda b: collate_fn(b, tokenizer))
    correct = total = 0
    examples = []  # (text, pred_label, true_label, gen_tail)

    pbar = tqdm(loader, desc="accuracy: 0.0000", leave=False)
    with torch.no_grad():
        for batch in pbar:
            batch = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outs = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            full = tokenizer.batch_decode(outs, skip_special_tokens=True)
            prompts = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)

            preds = []
            tails = []
            for f, p in zip(full, prompts):
                tail = f[len(p):]
                tails.append(tail)
                preds.append(classify_from_suffix(tail))

            y = batch["labels"].tolist()
            total += len(y)
            correct += sum(int(pi == yi) for pi, yi in zip(preds, y))
            pbar.set_description(f"accuracy: {correct/total:.4f}")

            if keep_examples and len(examples) < max_examples:
                for t, pr, gt, tail in zip(batch["texts"], preds, y, tails):
                    if len(examples) < max_examples:
                        examples.append((t, pr, gt, tail))

    acc = correct / max(1, total)
    return acc, examples


def save_examples(examples, path, header):
    label_map = {0: "negative", 1: "positive", 2: "neutral"}
    with open(path, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        for (txt, pred, gt, tail) in examples:
            f.write("\n" + "-"*80 + "\n")
            f.write(f"Text: {txt}\n")
            f.write(f"Pred: {label_map[pred]} | True: {label_map[int(gt)]}\n")
            f.write(f"Gen tail: {tail.strip()[:200]}\n")


def main():
    # 1) Model stack
    model, tok = load_stack()
    print_prompt_stats(model)

    # 2) Dataset
    print("🔹 Preparing evaluation splits...")
    ds_retain = build_eval_split(tok, "test_retain")
    ds_forget = build_eval_split(tok, "test_forget")

    # 3) Which checkpoints?
    iters = CHECKPOINT_ITERS if CHECKPOINT_ITERS else list_available_iters(SPUL_DIR)
    if not iters:
        raise RuntimeError("No prompt_checkpoint_iter_*.pt files found to evaluate.")

    # 4) Sweep
    print("\n=== SWEEP across ES checkpoints ===")
    results = []  # rows for CSV
    best = {"iter": None, "retain": -1.0, "forget": -1.0}

    for it in iters:
        print(f"\n--- Iter {it} ---")
        load_spul_prompt_checkpoint(model, SPUL_DIR, it)
        # quick sanity: prompt stats
        print_prompt_stats(model)

        retain_acc, _ = evaluate_split(model, tok, ds_retain)
        forget_acc, _ = evaluate_split(model, tok, ds_forget)
        print(f"iter {it} | retain_acc={retain_acc:.4f} | forget_acc={forget_acc:.4f}")

        results.append([it, retain_acc, forget_acc])

        # track "best" by retain accuracy (you can change criterion)
        if retain_acc > best["retain"]:
            best = {"iter": it, "retain": retain_acc, "forget": forget_acc}

    # 5) Save CSV
    with open(RESULTS_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iter", "retain_acc", "forget_acc"])
        w.writerows(results)
    print(f"\n📄 Wrote CSV: {RESULTS_CSV}")
    print(f"Best by retain accuracy: iter {best['iter']} | retain={best['retain']:.4f} | forget={best['forget']:.4f}")

    # 6) Dump examples for a particular checkpoint
    if EXAMPLE_CHECKPOINT is not None:
        print(f"\n=== Writing {EXAMPLE_COUNT} examples for iter {EXAMPLE_CHECKPOINT} ===")
        load_spul_prompt_checkpoint(model, SPUL_DIR, EXAMPLE_CHECKPOINT)
        retain_acc, retain_examples = evaluate_split(
            model, tok, ds_retain, keep_examples=True, max_examples=EXAMPLE_COUNT
        )
        header = f"Examples for iter {EXAMPLE_CHECKPOINT} (retain_acc={retain_acc:.4f})"
        save_examples(retain_examples, EXAMPLE_LOG_PATH, header)
        print(f"📝 Saved examples to {EXAMPLE_LOG_PATH}")


if __name__ == "__main__":
    main()
