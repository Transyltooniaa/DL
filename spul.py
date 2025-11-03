# Unlearn using SPUL

import os
import pickle
import datetime
import time
import random
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from numpy.random import default_rng
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
from torchinfo import summary

from datasets import load_dataset, concatenate_datasets
import evaluate
from peft import get_peft_model, TaskType, PromptEncoderConfig, PeftConfig, PeftModel
from transformers import AutoTokenizer, TrainerState, TrainerControl, AutoModelForCausalLM, Trainer, TrainingArguments, TrainerCallback
import nevergrad as ng

try:
    # TRL placement varies across versions
    from trl import DataCollatorForCompletionOnlyLM
except Exception:
    try:
        from trl.data import DataCollatorForCompletionOnlyLM
    except Exception:
        # Fallback collator if TRL doesn't provide one
        class DataCollatorForCompletionOnlyLM:
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

from utils import get_data_path, compute_metrics, preprocess_logits_for_metrics, get_logits_from_base_model, CustomCallback

POS_WEIGHT, NEG_WEIGHT = (1.0, 1.0)


def get_args():
    parser = ArgumentParser(description="LLM Unlearning using SPUL method")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        required=True,
        help="Name of dataset",
    )
    parser.add_argument(
        "--model_checkpoints",
        type=str,
        default=None,
        required=True,
        help="Path to checkpoints for base model to be unlearned",
    )
    parser.add_argument(
        "--logits_path",
        type=str,
        default=None,
        required=False,
        help="Path to save original logits to use for KL loss",
    )
    parser.add_argument(
        "--forget_size",
        type=float,
        default=1.0,
        required=False,
        help="relative size of forget set for ablation",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        required=False,
        help="Path to store the unlearned model",
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
        "--ptuning_num_tokens", type=int, default=30, help="Number of learnable tokens (p)"
    )
    parser.add_argument(
        "--ptuning_hidden_size", type=int, default=128, help="Number of hidden dimensions for prompt encoder"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5, help="weight for retain CE loss"
    )
    parser.add_argument(
        "--beta", type=float, default=0.5, help="weight for KL loss"
    )

    # Optimizer choices: backprop (existing Trainer) or nevergrad (ES)
    parser.add_argument(
        "--optimizer",
        type=str,
        default="backprop",
        choices={"backprop", "nevergrad"},
        help="Optimizer to use for prompt optimization: 'backprop' or 'nevergrad' (ES)"
    )
    # ES hyperparameters (used when --optimizer nevergrad)
    parser.add_argument("--es_pop_size", type=int, default=32, help="Population size for ES (nevergrad)")
    parser.add_argument("--es_iters", type=int, default=200, help="Number of ES iterations (ask/tell cycles)")
    parser.add_argument("--es_batch_size", type=int, default=8, help="Batch size to evaluate candidates when computing fitness")
    parser.add_argument("--es_seed", type=int, default=42, help="Random seed for the ES optimizer")
    parser.add_argument("--es_eval_batches", type=int, default=4, help="Number of eval batches to run per candidate (limits cost)")
    parser.add_argument("--es_normalize", action='store_true', help="Rank-normalize candidate losses before telling optimizer (improves robustness)")
    parser.add_argument("--es_opt", type=str, default='CMA', choices={'CMA', 'OnePlusOne'}, help="Which Nevergrad optimizer to use")

    arguments = parser.parse_args()
    return arguments


def get_ptuning_model(model_checkpoints, max_length, num_tokens, prompt_encoder_hidden_size):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftConfig, PeftModel, get_peft_model, TaskType, PromptEncoderConfig
    import os, sys

    lora_peft_model_id = model_checkpoints
    lora_config = PeftConfig.from_pretrained(lora_peft_model_id)
    base_id = lora_config.base_model_name_or_path  # "facebook/opt-1.3b" in your case

    # Helpful logs
    print(f"[SPUL] Loading base model: {base_id}")
    print(f"[SPUL] LoRA adapter path: {lora_peft_model_id}")

    # Try multiple loading strategies
    last_err = None
    for use_safetensors in (True, False):
        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                base_id,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
                use_safetensors=use_safetensors,   # try safetensors then bin
            ).to("cuda")
            print(f"[SPUL] Loaded base model (use_safetensors={use_safetensors})")
            break
        except Exception as e:
            last_err = e
            print(f"[SPUL] Failed to load with use_safetensors={use_safetensors}: {e}")

    if "base_model" not in locals():
        # Helpful guidance before failing
        raise RuntimeError(
            "Failed to load base model '{base_id}'.\n"
            "Checklist:\n"
            " - Are you online & `huggingface-cli login`-ed?\n"
            " - Is there an accidental local folder named 'facebook/opt-1.3b'?\n"
            " - Does the HF repo have pytorch_model.bin (if no safetensors)?\n"
            f"Original error: {last_err}"
        )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_id,
        truncation=True,
        padding=True,
        max_length=max_length,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Attach and merge LoRA
    print("[SPUL] Loading LoRA adapter and merging...")
    lora_model = PeftModel.from_pretrained(
        base_model,
        lora_peft_model_id,
        is_trainable=False,
    )
    lora_model = lora_model.merge_and_unload()

    original_model = deepcopy(lora_model)

    # Add SPUL prompt (p-tuning)
    ptuning_peft_config = PromptEncoderConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=num_tokens,
        encoder_hidden_size=prompt_encoder_hidden_size
    )
    model = get_peft_model(lora_model, ptuning_peft_config)
    return model, original_model, tokenizer

 


def get_unlearn_dataset_and_collator(
        data_path,
        tokenizer,
        forget_size=1.0,
        add_prefix_space=True,
        max_length=1024,
        truncation=True
):
    prompt_template = lambda text, label: f"""### Text: {text}\n\n### Question: What is the sentiment of the given text?\n\n### Sentiment: {label}"""

    def _preprocessing_sentiment(examples):
        return tokenizer(prompt_template(examples['text'], examples['label_text']), truncation=truncation, max_length=max_length )
    
    
    response_template = "\n### Sentiment:"
    response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)
    data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

    data = load_dataset(data_path)

    neutral_label = 'neutral'

    # For ablation, sample smaller size of train_forget
    if forget_size < 1.0:
        train_forget_size = int(forget_size * data['train_forget'].num_rows)
        rng = default_rng(seed=42)
        train_forget_indx = rng.choice(data['train_forget'].num_rows, size=train_forget_size, replace=False)
        data['train_forget'] = data['train_forget'].select(train_forget_indx)

    # Sample random answer for forget samples
    train_forget_flip = deepcopy(data['train_forget'])
    train_forget_flip = train_forget_flip.map(lambda item: {"label_text": neutral_label})
    data['train_forget_flip'] = train_forget_flip

    data['train_forget_flip'] = data['train_forget_flip'].map(lambda item: {"is_forget": 1})
    data['train_retain'] = data['train_retain'].map(lambda item: {"is_forget": 0})    
    data['train'] = concatenate_datasets([data['train_retain'], data['train_forget_flip']])
    
    del data['train_forget_flip']
    data['train_retain'] = data['train_retain'].remove_columns('is_forget')

    data['train'] = data['train'].map(lambda item, idx: {"index": idx}, with_indices=True)

    col_to_delete = ['text', 'label', 'label_text']
    data = data.map(_preprocessing_sentiment, batched=False)
    data = data.remove_columns(col_to_delete)
    data.set_format("torch")        

    print(data)
    # print(data['train']['text'][:10])

    return data, data_collator

def get_unlearning_loss_trainer():
    class UnlearningTrainer(Trainer):
        def __init__(self, original_logits, num_virtual_tokens, alpha, beta, **kwargs):
            super().__init__(**kwargs)
            self.name = 'SPUL'
            self.num_virtual_tokens = num_virtual_tokens
            self.original_logits = original_logits
            self.alpha=alpha
            self.beta=beta

        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.get("labels")
            if "is_forget" not in inputs or "index" not in inputs:
                outputs = model(**inputs)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                return (loss, outputs) if return_outputs else loss

            is_forget = inputs.pop("is_forget")          # [B]
            indices   = inputs.pop("index")              # [B]

            outputs = model(**inputs)
            logits  = outputs.get("logits")              # [B, T, V]

            # concat prefix -100s for virtual tokens (align with prompt length p)
            prefix = torch.full((labels.size(0), self.num_virtual_tokens), -100, device=labels.device)
            labels_cat = torch.cat((prefix, labels), dim=1)  # [B, T+p]

            shift_logits = logits[:, :-1, :].contiguous()    # [B, T-1, V]
            shift_labels = labels_cat[:, 1:].contiguous()    # [B, T-1]

            # masks
            fmask = (is_forget > 0)
            rmask = ~fmask

            ce_tok = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)

            def token_ce(logits_bt, labels_bt):
                # logits_bt: [B, T, V], labels_bt: [B, T]
                if logits_bt.numel() == 0:
                    return torch.tensor(0.0, device=logits_bt.device)
                flat_logits = logits_bt.view(-1, logits_bt.size(-1))
                flat_labels = labels_bt.view(-1)
                losses = ce_tok(flat_logits, flat_labels)              # [B*T]
                valid = (flat_labels != -100)
                denom = valid.sum().clamp_min(1)
                return (losses[valid].sum() / denom)

            # -------- RETAIN: CE over answer tokens (token-normalized) --------
            retain_ce = torch.tensor(0.0, device=logits.device)
            if rmask.any():
                retain_ce = token_ce(shift_logits[rmask], shift_labels[rmask])

            # -------- FORGET: uncertainty via KL to uniform over answer tokens --------
            forget_unc = torch.tensor(0.0, device=logits.device)
            if fmask.any():
                f_logits = shift_logits[fmask]     # [F, T, V]
                f_labels = shift_labels[fmask]     # [F, T]
                ans_mask = (f_labels != -100)      # valid answer positions
                if ans_mask.any():
                    V = f_logits.size(-1)
                    flat_logits = f_logits[ans_mask]                 # [N_valid, V]
                    uniform = torch.full_like(flat_logits, 1.0 / V)  # target
                    kld = torch.nn.KLDivLoss(reduction='batchmean')  # token-averaged KL
                    forget_unc = kld(torch.log_softmax(flat_logits, dim=-1),
                                    torch.softmax(uniform, dim=-1))

            # -------- RETAIN: KL to original on FIRST answer token only --------
            rtn_kl = torch.tensor(0.0, device=logits.device)
            if self.beta > 0 and rmask.any():
                r_logits = shift_logits[rmask]     # [R, T, V]
                r_labels = shift_labels[rmask]     # [R, T]
                first_logits = []
                target_logits = []

                r_indices = indices[rmask]         # original indices aligned to retain rows

                for b in range(r_labels.size(0)):
                    row = r_labels[b]
                    idxs = torch.nonzero(row != -100, as_tuple=False).flatten()
                    if idxs.numel() == 0:
                        continue
                    first_pos = idxs[0].item()
                    first_logits.append(r_logits[b, first_pos, :])
                    vec = torch.tensor(self.original_logits[int(r_indices[b])], device=logits.device)
                    target_logits.append(vec)

                if len(first_logits) > 0:
                    p = torch.stack(first_logits, dim=0)
                    q = torch.stack(target_logits, dim=0)
                    kld = torch.nn.KLDivLoss(reduction='batchmean')
                    rtn_kl = kld(torch.log_softmax(p, dim=-1), torch.softmax(q, dim=-1))

            # total
            loss = forget_unc + self.alpha * retain_ce + self.beta * rtn_kl
            return (loss, outputs) if return_outputs else loss



    return UnlearningTrainer


def _find_prompt_param_name(model, num_virtual_tokens=None):
    """
    Try to heuristically locate the prompt-embedding parameter name in the PEFT-wrapped model.
    Preference: parameter whose first dimension equals num_virtual_tokens if provided, else look for 'prompt' in name.
    Returns the parameter name (string) and its tensor.
    """
    # First try to find by name
    for n, p in model.named_parameters():
        if 'prompt' in n or 'virtual' in n or 'prompt_encoder' in n:
            return n, p

    # Fallback: match first dim
    if num_virtual_tokens is not None:
        for n, p in model.named_parameters():
            try:
                if p.shape[0] == num_virtual_tokens:
                    return n, p
            except Exception:
                continue

    # Last resort: return first parameter that requires_grad
    for n, p in model.named_parameters():
        if p.requires_grad:
            return n, p

    raise RuntimeError('Unable to find prompt parameter in model')


def _set_prompt_from_vector(model, param_name, vector, device):
    """Set the parameter named param_name in model from a flat numpy vector."""
    with torch.no_grad():
        # find the parameter
        for n, p in model.named_parameters():
            if n == param_name:
                param = p
                break
        else:
            raise RuntimeError(f'Parameter {param_name} not found')

        shape = param.shape
        tensor = torch.from_numpy(vector.reshape(shape)).to(device)
        param.data.copy_(tensor)


def _compute_spul_loss_for_batch(model, batch, original_logits, num_virtual_tokens, alpha, beta, device):
    """Compute SPUL loss used by ES: token-normalized CE for retain,
       KL-to-uniform for forget (answer tokens), and first-token KL to original on retain."""
    batch = {k: v.to(device) for k, v in batch.items()}
    labels = batch.get('labels')

    is_forget = batch.pop('is_forget') if 'is_forget' in batch else None
    indices   = batch.pop('index') if 'index' in batch else None

    model.eval()
    with torch.no_grad():
        outputs = model(**batch)
    logits = outputs.get('logits')  # [B, T, V]

    prefix = torch.full((len(labels), num_virtual_tokens), -100, device=labels.device)
    labels_cat = torch.cat((prefix, labels), dim=1)

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels_cat[:, 1:].contiguous()

    if is_forget is None or indices is None:
        # fallback
        loss = outputs['loss'] if isinstance(outputs, dict) and 'loss' in outputs else shift_logits.mean()
        return loss.detach().cpu()

    fmask = (is_forget > 0)
    rmask = ~fmask

    ce_tok = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)

    def token_ce(logits_bt, labels_bt):
        if logits_bt.numel() == 0:
            return torch.tensor(0.0, device=logits_bt.device)
        flat_logits = logits_bt.view(-1, logits_bt.size(-1))
        flat_labels = labels_bt.view(-1)
        losses = ce_tok(flat_logits, flat_labels)
        valid = (flat_labels != -100)
        denom = valid.sum().clamp_min(1)
        return (losses[valid].sum() / denom)

    # retain CE
    retain_ce = torch.tensor(0.0, device=logits.device)
    if rmask.any():
        retain_ce = token_ce(shift_logits[rmask], shift_labels[rmask])

    # forget uncertainty (KL to uniform)
    forget_unc = torch.tensor(0.0, device=logits.device)
    if fmask.any():
        f_logits = shift_logits[fmask]
        f_labels = shift_labels[fmask]
        ans_mask = (f_labels != -100)
        if ans_mask.any():
            V = f_logits.size(-1)
            flat_logits = f_logits[ans_mask]                 # [N_valid, V]
            uniform = torch.full_like(flat_logits, 1.0 / V)
            kld = torch.nn.KLDivLoss(reduction='batchmean')
            forget_unc = kld(torch.log_softmax(flat_logits, dim=-1),
                             torch.softmax(uniform, dim=-1))

    # retain first-token KL to original
    rtn_kl = torch.tensor(0.0, device=logits.device)
    if beta > 0 and rmask.any():
        r_logits = shift_logits[rmask]
        r_labels = shift_labels[rmask]

        first_logits = []
        target_logits = []
        r_indices = indices[rmask]

        for b in range(r_labels.size(0)):
            row = r_labels[b]
            idxs = torch.nonzero(row != -100, as_tuple=False).flatten()
            if idxs.numel() == 0:
                continue
            first_pos = idxs[0].item()
            first_logits.append(r_logits[b, first_pos, :])
            vec = torch.tensor(original_logits[int(r_indices[b])], device=logits.device)
            target_logits.append(vec)

        if len(first_logits) > 0:
            p = torch.stack(first_logits, dim=0)
            q = torch.stack(target_logits, dim=0)
            kld = torch.nn.KLDivLoss(reduction='batchmean')
            rtn_kl = kld(torch.log_softmax(p, dim=-1), torch.softmax(q, dim=-1))

    total = forget_unc + alpha * retain_ce + beta * rtn_kl
    return total.detach().cpu()

def run_nevergrad_es(model, tokenizer, dataset, collator, original_logits, args):
    """Run Nevergrad ES to optimize the prompt embeddings on GPU."""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Find the prompt parameter
    param_name, param = _find_prompt_param_name(model, num_virtual_tokens=args.ptuning_num_tokens)
    param_tensor = param.detach().cpu().numpy()
    param_shape = param_tensor.shape
    dim = int(np.prod(param_shape))
    init = param_tensor.reshape(-1).astype(np.float32)
    print(f"Found prompt parameter: '{param_name}' with shape={param_shape}, flattened_dim={dim}")

    # Initialize optimizer
    parametrization = ng.p.Array(init=init)
    budget = int(max(1, args.es_iters * args.es_pop_size))
    print(f"Initializing Nevergrad optimizer '{args.es_opt}' with budget={budget}, pop_size={args.es_pop_size}")

    if args.es_opt == 'CMA':
        optimizer = ng.optimizers.CMA(
            parametrization=parametrization,
            budget=budget,
            num_workers=args.es_pop_size,
        )
    else:
        optimizer = ng.optimizers.OnePlusOne(
            parametrization=parametrization,
            budget=budget,
            num_workers=args.es_pop_size
        )

    eval_loader = DataLoader(dataset['train'], collate_fn=collator, batch_size=args.es_batch_size)

    best_loss = float('inf')
    best_vector = init.copy()

    np.random.seed(args.es_seed)
    random.seed(args.es_seed)
    torch.manual_seed(args.es_seed)

    model.to(device)
    model.eval()

    for iteration in range(args.es_iters):
        best_this_iter = None

        for i in range(args.es_pop_size):
            cand = optimizer.ask()                           # 1) ask
            vec = np.array(cand.value, dtype=np.float32)

            _set_prompt_from_vector(model, param_name, vec, device)

            total_loss = 0.0
            n_batches = 0
            for j, batch in enumerate(eval_loader):
                if j >= args.es_eval_batches:
                    break
                total_loss += float(_compute_spul_loss_for_batch(
                    model, batch, original_logits,
                    args.ptuning_num_tokens, args.alpha, args.beta, device
                ))
                n_batches += 1

            fitness = total_loss / max(1, n_batches)

            optimizer.tell(cand, float(fitness))             # 2) tell immediately

            # track best
            if (best_this_iter is None) or (fitness < best_this_iter):
                best_this_iter = fitness
            if fitness < best_loss:
                best_loss = fitness
                best_vector = vec.copy()

        # checkpoint/logging
        checkpoint_every = max(1, args.es_iters // 10)
        if (iteration + 1) % checkpoint_every == 0 or (iteration + 1) <= 5:
            print(f"ES iter {iteration+1}/{args.es_iters}, best_loss={best_loss}")
            os.makedirs(args.output_path, exist_ok=True)
            ckpt_path = os.path.join(args.output_path, f'prompt_checkpoint_iter_{iteration+1}.pt')
            torch.save(torch.from_numpy(best_vector.reshape(param_shape)), ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")


    # Finalize
    _set_prompt_from_vector(model, param_name, best_vector.astype(np.float32), device)
    model.train()
    model.save_pretrained(args.output_path)
    print(f"✅ Saved ES-optimized prompt weights to {args.output_path}")


def main(args):
    # Sync wandb
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["WANDB_LOG_MODEL"] = "all"  # log your models

    if 'llama-2-7b' in args.model_checkpoints.lower():
        model_name = 'llama-2-7b-hf'
    elif 'llama-2-13b' in args.model_checkpoints.lower():
        model_name = 'llama-2-13b-hf'
    elif 'opt-1.3b' in args.model_checkpoints.lower():
        model_name = 'opt-1.3b'

    os.environ["WANDB_PROJECT"] = f'spul_{model_name}_{args.dataset.lower()}' 

    data_path = get_data_path(args.dataset)
    
    model, original_model, tokenizer = get_ptuning_model(
        args.model_checkpoints,
        args.max_length,
        args.ptuning_num_tokens,
        args.ptuning_hidden_size
    )

    dataset, collator = get_unlearn_dataset_and_collator(
        data_path,
        # args.model_checkpoints,
        tokenizer=tokenizer,
        max_length=args.max_length,
        forget_size=args.forget_size,
        add_prefix_space=True,
        truncation=True,
    )

    if args.logits_path is None:
        args.logits_path = f'saved_logits/{model_name}_{args.dataset.lower()}-{args.forget_size}.pkl'

    if not os.path.exists(args.logits_path):
        print('Saving original logits from base model')
        original_logits = get_logits_from_base_model(original_model, collator, dataset)

        new_original_logits = {}
        for k, v in original_logits.items():
            idx = int(k.item()) if hasattr(k, "item") else int(k)
            if torch.is_tensor(v):
                new_original_logits[idx] = v.detach().cpu().numpy()
            else:
                new_original_logits[idx] = np.array(v)

        os.makedirs(os.path.dirname(args.logits_path), exist_ok=True)
        with open(args.logits_path, 'wb') as f:
            pickle.dump(new_original_logits, f, protocol=pickle.HIGHEST_PROTOCOL)

        print('Completed saving logits from base model')

    with open(args.logits_path, 'rb') as f:
        print('Loading original logits from base model')
        original_logits = pickle.load(f)

    if args.output_path is None:
        args.output_path = f'unlearn_checkpoints/spul_{model_name}_{args.dataset.lower()}-{args.forget_size}_{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'
        
        if not os.path.exists(args.output_path):
            os.mkdir(args.output_path)

        args_dict = args.__dict__.copy()
        args_dict['model_name'] = args.model_checkpoints
        with open(os.path.join(args.output_path, 'arguments.txt'), 'w') as f:
            for k, v in args_dict.items():
                f.write(f'{k}: {v}\n')
                
    training_args = TrainingArguments(
        output_dir=args.output_path,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        eval_strategy="no",
        save_strategy="no",
        group_by_length=True,
        gradient_checkpointing=True,
        fp16=True,
        report_to="wandb",
        run_name=f'lr={args.lr}_alpha={args.alpha}_beta={args.beta}_numtokens={args.ptuning_num_tokens}',
        max_grad_norm=0.3,
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_train_retain_loss"
    )

    if args.set_pad_id:
        model.config.pad_token_id = model.config.eos_token_id

    # move model to GPU device
    if model.device.type != 'cuda':
        model = model.to('cuda')

    # If using ES optimizer (nevergrad), run the ES loop instead of backprop-based Trainer
    if args.optimizer == 'nevergrad':
        print('Running Nevergrad ES optimizer for prompt optimization (this may take a while)')
        run_nevergrad_es(model, tokenizer, dataset, collator, original_logits, args)
        return

    # Otherwise fall back to backprop-based Trainer defined earlier
    custom_loss = get_unlearning_loss_trainer()

    trainer = custom_loss(
        model=model,
        original_logits=original_logits,
        num_virtual_tokens=args.ptuning_num_tokens,
        alpha=args.alpha,
        beta=args.beta,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=dataset['train'],
        eval_dataset={"train_retain": dataset['train_retain'],
                      "train_forget": dataset['train_forget']},
        data_collator=collator,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_metrics
    )

    trainer.add_callback(CustomCallback(trainer))
    start = time.perf_counter()
    trainer.train()
    runtime = (time.perf_counter()-start)
    print(runtime)


if __name__ == "__main__":
    args = get_args()
    main(args)