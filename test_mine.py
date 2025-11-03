import os
import torch
import json
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from transformers.utils import logging as hf_logging





# Silence noisy warnings
hf_logging.set_verbosity_error()
os.environ["TRANSFORMERS_NO_SAFE_TORCH_LOAD"] = "1"

# ================================================================
# CONFIGURATION
# ================================================================
BASE_MODEL = "facebook/opt-1.3b"
QLORA_DIR = os.path.expanduser(
    "~/DeepLearning/Abhinav/qlora_checkpoints/opt-1.3b-qlora-sst2/final_model"
)
SPUL_DIR = "./unlearn_checkpoints/spul_opt1.3b_sst2_nes"
DATASET_NAME = "karuna-bhaila/Unlearning_SST2v3"
MAX_LENGTH = 384
BATCH_SIZE = 8
MAX_NEW_TOKENS = 3
SAMPLE_LOG_PATH = "sample_outputs.txt"

# ================================================================
# LOAD TOKENIZER + MODEL STACK
# ================================================================
print("🔹 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"  # required for decoder-only models

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

# ================================================================
# VERIFY ADAPTER LOAD
# ================================================================
print("Checking prompt embedding statistics:")
for n, p in model.named_parameters():
    if "prompt" in n or "virtual" in n:
        print(f"  {n} | mean={p.data.mean().item():.6f}, std={p.data.std().item():.6f}")

print("\nIf means/stds are near zero (like 0.0000), SPUL adapter might not have loaded.\n")

# ================================================================
# LOAD DATA
# ================================================================
EVAL_TEMPLATE = (
    "### Text: {text}\n\n### Question: What is the sentiment of the given text?\n\n### Sentiment:"
)

def build_eval_split(split):
    raw = load_dataset(DATASET_NAME, split=split)

    def _prep(ex):
        prompt = EVAL_TEMPLATE.format(text=ex["text"]).strip()
        enc = tokenizer(prompt, truncation=True, padding=False, max_length=MAX_LENGTH)
        ex_out = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "label": ex["label"],
        }
        # ✅ keep a copy of original text separately
        ex_out["orig_text"] = ex["text"]
        return ex_out

    ds = raw.map(_prep, remove_columns=raw.column_names)
    # only numeric columns become tensors; text stays python str
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    return ds

print("🔹 Preparing evaluation splits...")
ds_retain = build_eval_split("test_retain")
ds_forget = build_eval_split("test_forget")

# ================================================================
# COLLATE + INFERENCE
# ================================================================

def collate_fn(batch):
    # ⚙️ handle both dict and DatasetRow cases
    texts = [b.get("orig_text", "") for b in batch]
    labels = torch.tensor([int(b["label"]) for b in batch], dtype=torch.long)
    # pad variable-length token sequences
    enc = tokenizer.pad(
        [{"input_ids": b["input_ids"], "attention_mask": b["attention_mask"]} for b in batch],
        return_tensors="pt",
        padding=True,
    )
    enc["labels"] = labels
    enc["texts"] = texts
    return enc

def run_eval(split_name, ds):
    loader = DataLoader(ds, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    correct = total = 0
    label_map = {0: "negative", 1: "positive", 2: "neutral"}

    with open(SAMPLE_LOG_PATH, "a") as f:
        f.write(f"\n=== {split_name} Samples ===\n")

    pbar = tqdm(loader, desc=f"{split_name} accuracy: 0.000")
    with torch.no_grad():
        for batch in pbar:
            batch = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outs = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            full_texts = tokenizer.batch_decode(outs, skip_special_tokens=True)
            prompt_texts = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
            preds = []
            for full, prompt in zip(full_texts, prompt_texts):
                suffix = full[len(prompt):].strip().lower()
                if suffix.startswith("positive"):
                    preds.append(1)
                elif suffix.startswith("negative"):
                    preds.append(0)
                elif suffix.startswith("neutral"):
                    preds.append(2)
                else:
                    if "positive" in suffix:
                        preds.append(1)
                    elif "negative" in suffix:
                        preds.append(0)
                    elif "neutral" in suffix:
                        preds.append(2)
                    else:
                        preds.append(2)
            y = batch["labels"].tolist()
            total += len(y)
            correct += sum(int(pi == yi) for pi, yi in zip(preds, y))
            acc = correct / total
            pbar.set_description(f"{split_name} accuracy: {acc:.4f}")

            # Save some examples
            with open(SAMPLE_LOG_PATH, "a") as f:
                for text, pred, label, full in zip(batch["texts"], preds, y, full_texts[:3]):
                    f.write(
                        f"\nText: {text[:80]}...\nPred: {label_map[pred]}, True: {label_map[label]}\n"
                        f"Gen: {full[-80:]}\n"
                    )

    print(f"\n=== {split_name} RESULTS ===")
    print(f"Final accuracy: {acc:.4f} ({correct}/{total})\n")

# ================================================================
# RUN
# ================================================================
if __name__ == "__main__":
    print("=== Evaluating on test_retain ===")
    run_eval("test_retain", ds_retain)
    print("=== Evaluating on test_forget ===")
    run_eval("test_forget", ds_forget)
