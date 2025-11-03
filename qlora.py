# Training script to fine-tune a pre-train LLM with QLoRA using HuggingFace.

import os
import sys
import time
from argparse import ArgumentParser
from copy import deepcopy
import evaluate
import numpy as np

import torch
from torchinfo import summary
import inspect

from datasets import load_dataset, concatenate_datasets
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoTokenizer, TrainerState, TrainerControl, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import TrainingArguments, TrainerCallback
try:
    # Newer TRL versions expose DataCollatorForCompletionOnlyLM at top-level
    from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
except Exception:
    # Import SFTTrainer (may be top-level); if DataCollator isn't available,
    # provide a small local fallback that tokenizes text and returns input_ids
    # and labels for causal LM training. This fallback is intentionally
    # conservative and works for smoke tests and small runs.
    try:
        from trl import SFTTrainer
    except Exception:
        # If trl is not installed at all, raise a clear error
        raise ImportError("trl (TRL) is required for QLoRA training. Install via `pip install trl`.")

    class DataCollatorForCompletionOnlyLM:
        """Lightweight fallback collator.

        This simple collator tokenizes a list of examples (expects each example
        to have a `text` field) and returns tensors suitable for causal LM
        training: `input_ids`, `attention_mask` (if available), and `labels`
        (copy of input_ids).
        """
        def __init__(self, response_template_ids=None, tokenizer=None, max_length=1024):
            self.response_template_ids = response_template_ids
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __call__(self, features):
            texts = [f.get('text', '') for f in features]
            enc = self.tokenizer(
                texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            input_ids = enc['input_ids']
            attention_mask = enc.get('attention_mask', None)
            labels = input_ids.clone()
            batch = {'input_ids': input_ids, 'labels': labels}
            if attention_mask is not None:
                batch['attention_mask'] = attention_mask
            return batch

from utils import get_data_path, compute_metrics, preprocess_logits_for_metrics, CustomCallback

POS_WEIGHT, NEG_WEIGHT = (1.0, 1.0)

def get_args():
    parser = ArgumentParser(description="Fine-tune an LLM model with QLoRA")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        required=True,
        help="name of dataset",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        required=True,
        help="Checkpoints to path of the pre-trained LLM to fine-tune",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        required=False,
        help="Path to store the fine-tuned model",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        required=False,
        help="Maximum length of the input sequences",
    )
    parser.add_argument(
        "--set_pad_id",
        action="store_true",
        help="Set the id for the padding token, needed by models such as Mistral-7B",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate for training"
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=32, help="Train batch size"
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=32, help="Eval batch size"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=10, help="Number of epochs"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.001, help="Weight decay"
    )
    parser.add_argument(
        "--lora_rank", type=int, default=16, help="Lora rank"
    )
    parser.add_argument(
        "--lora_alpha", type=float, default=64, help="Lora alpha"
    )
    parser.add_argument(
        "--lora_dropout", type=float, default=0.1, help="Lora dropout"
    )
    parser.add_argument(
        "--lora_bias",
        type=str,
        default='none',
        choices={"lora_only", "none", 'all'},
        help="Layers to add learnable bias"
    )

    arguments = parser.parse_args()
    return arguments

def get_lora_model(model_checkpoints, rank=4, alpha=16, lora_dropout=0.1, bias='none'):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    # Try loading with device_map='auto' first (preferred). Some combinations
    # of transformers/accelerate/bitsandbytes raise a ValueError when an
    # internal `.to()` is attempted on 4-bit/8-bit models; if that happens,
    # retry without device_map. If both attempts fail, fall back to loading
    # the model without bitsandbytes quantization (may use much more memory).
    try:
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_checkpoints,
            device_map="auto",
            use_safetensors=True,
            quantization_config=bnb_config,
            trust_remote_code=True,
        )
    except ValueError as e:
        print(f"Warning: load with device_map='auto' failed: {e}\nRetrying without device_map to avoid calling .to() on a quantized model.")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=model_checkpoints,
                device_map=None,
                use_safetensors=True,
                quantization_config=bnb_config,
                trust_remote_code=True,
            )
        except ValueError as e2:
            print(f"Warning: retry without device_map also failed: {e2}\nFalling back to loading the model without bitsandbytes quantization. This may require much more GPU/CPU memory.")
            # Final fallback: load without quantization
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=model_checkpoints,
                device_map="auto",
                use_safetensors=True,
                trust_remote_code=True,
            )
    # Check for meta tensors (these indicate the model was created with
    # init_empty_weights and not fully materialized on real devices). If
    # present, moving the model to CUDA with `.to()` will fail. In that
    # case, surface a clear error with actionable fixes instead of
    # continuing and letting Trainer crash later.
    any_meta = any(getattr(p, 'is_meta', False) for p in model.parameters())
    if any_meta:
        msg = (
            "The loaded model contains tensors on the 'meta' device (uninitialized).\n"
            "This typically happens when transformers/accelerate instantiate the model with "
            "init_empty_weights and then expect to dispatch it to devices. Your environment "
            "triggered a fallback load that left parameters on 'meta'.\n\n"
            "Recommended fixes:\n"
            " 1) Install/upgrade bitsandbytes and accelerate so 4-bit quantization can be used:\n"
            "      pip install -U bitsandbytes accelerate transformers safetensors peft trl\n\n"
            " 2) If you don't have a compatible CUDA/bitsandbytes setup or limited GPU memory, "
            "try a smaller model for the smoke test (e.g., facebook/opt-1.3b):\n"
            "      python qlora.py --dataset sst2 --model_name facebook/opt-1.3b --output_path qlora_checkpoints/opt1.3b-smoke\n\n"
            " 3) If you must load full-precision models, ensure you have enough GPU/CPU memory and follow "
            "the transformers/accelerate guidance for large-model loading.\n\n"
            "After applying one of the fixes above, re-run the command. If you need help with specific "
            "install errors (bitsandbytes/accelerate), paste the error and I will help."
        )
        raise RuntimeError(msg)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoints)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if model_checkpoints == 'mistralai/Mistral-7B-v0.1' or model_checkpoints == 'meta-llama/Llama-2-7b-hf':
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=rank, lora_alpha=alpha, lora_dropout=lora_dropout, bias=bias,
            target_modules=[
                "q_proj",
                "v_proj",
            ],
        )
    else:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=rank, lora_alpha=alpha, lora_dropout=lora_dropout, bias=bias,
        )

    return model, tokenizer, peft_config


def get_unlearn_dataset_and_collator(
        data_path,
        tokenizer,
        add_prefix_space=True,
        max_length=1024,
        truncation=True
):
    prompt_template = lambda text, label: f"""### Text: {text}\n\n### Question: What is the sentiment of the given text?\n\n### Sentiment: {label}"""

    # Tokenize inputs
    def _preprocessing_sentiment(examples):
        return {"text": prompt_template(examples['text'], examples['label_text'])}

    response_template = "\n### Sentiment:"
    response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]

    data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

    data = load_dataset(data_path)

    data = data.map(_preprocessing_sentiment, batched=False)
    data = data.remove_columns(['label', 'label_text'])
    data.set_format("torch")

    print(data)

    return data, data_collator


def main(args):
    if 'llama-2-7b' in args.model_name.lower():
        model_name = 'llama-2-7b-hf'
    elif 'llama-2-13b' in args.model_name.lower():
        model_name = 'llama-2-13b-hf'
    elif 'opt-1.3b' in args.model_name.lower():
        model_name = 'opt-1.3b'
    else:
        raise NotImplementedError
    
    # Sync to wandb
    os.environ["WANDB_LOG_MODEL"] = "all"  # log your models
    os.environ["WANDB_PROJECT"] = f'qlora_{model_name.lower()}_{args.dataset.lower()}'  # log to your project
    
    data_path = get_data_path(args.dataset)

    if args.output_path is None:
        args.output_path = f'qlora_checkpoints/{model_name.lower()}-hf-qlora-{args.dataset.lower()}'

        if not os.path.exists(args.output_path):
            os.mkdir(args.output_path)
        with open(os.path.join(args.output_path, 'arguments.txt'), 'w') as f:
            for k, v in args.__dict__.items():
                f.write(f'{k}: {v}\n')

    # Initialize models and collator
    model, tokenizer, lora_config = get_lora_model(
        args.model_name,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias
    )

    dataset, collator = get_unlearn_dataset_and_collator(
        data_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
        add_prefix_space=True,
        truncation=True,
    )

    # Build TrainingArguments in a way that's robust to different transformers versions
    ta_kwargs = dict(
        output_dir=args.output_path,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        evaluation_strategy="no",
        save_strategy="no",
        group_by_length=True,
        load_best_model_at_end=True,
        gradient_checkpointing=True,
        fp16=True,
        report_to="wandb",
        run_name=f'lr={args.lr}',
        max_grad_norm=0.3,
        metric_for_best_model="eval_test_loss",
    )

    # Filter kwargs to only those accepted by the installed transformers.TrainingArguments
    try:
        sig = inspect.signature(TrainingArguments.__init__)
        valid_keys = set(sig.parameters.keys())
        valid_keys.discard('self')
        safe_kwargs = {k: v for k, v in ta_kwargs.items() if k in valid_keys}
    except Exception:
        # Fallback: if signature introspection fails, pass the original dict
        safe_kwargs = ta_kwargs

    training_args = TrainingArguments(**safe_kwargs)

    summary(model)

    if args.set_pad_id:
        model.config.pad_token_id = model.config.eos_token_id

    # move model to GPU device
    # For bitsandbytes 4-bit/8-bit quantized models `.to()` is not supported
    # because the model is already dispatched and cast by the HF/BNB loader.
    # Attempt to move only if safe; otherwise skip with an informative message.
    try:
        if hasattr(model, 'device') and getattr(model.device, 'type', None) != 'cuda':
            model = model.to('cuda')
    except Exception as e:
        print(f"Skipping model.to('cuda') (likely a 4/8-bit quantized model): {e}")

    # Build kwargs for SFTTrainer and filter to supported args to be robust
    sft_kwargs = dict(
        model=model,
        args=training_args,
        peft_config=lora_config,
        dataset_text_field='text',
        max_seq_length=args.max_length,
        tokenizer=tokenizer,
        train_dataset=concatenate_datasets([dataset['train_retain'], dataset['train_forget']]),
        eval_dataset={"test": concatenate_datasets([dataset['test_retain'], dataset['test_forget']])},
        data_collator=collator,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_metrics,
    )

    try:
        sft_sig = inspect.signature(SFTTrainer.__init__)
        sft_valid = set(sft_sig.parameters.keys())
        sft_valid.discard('self')
        sft_safe_kwargs = {k: v for k, v in sft_kwargs.items() if k in sft_valid}
    except Exception:
        sft_safe_kwargs = sft_kwargs

    trainer = SFTTrainer(**sft_safe_kwargs)
    trainer.add_callback(CustomCallback(trainer))
    start = time.perf_counter()
    trainer.train()
    runtime = (time.perf_counter()-start)
    print(runtime)


if __name__ == "__main__":
    args = get_args()
    main(args)
