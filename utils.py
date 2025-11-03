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
from copy import deepcopy
try:
    # TRL placement varies across versions
    from trl import DataCollatorForCompletionOnlyLM
except Exception:
    try:
        from trl.data import DataCollatorForCompletionOnlyLM
    except Exception:
        # Fallback lightweight collator for environments without TRL collator.
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
                # Provide is_forget/index fields default empty to keep downstream code compatible
                return batch

def compute_metrics(eval_pred):
    """
    Compute evaluation metrics (F1, Accuracy, Precision, Recall)
    for model predictions.
    """
    import evaluate
    f1_metric = evaluate.load("f1")
    accuracy_metric = evaluate.load("accuracy")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")

    logits, labels = eval_pred
    predictions = logits[:, :-1]
    labels = labels[:, 1:]
    check_labels = labels != -100

    last_token_predictions = []
    last_token_labels = []

    for idx in range(len(predictions)):
        last_token_predictions.append(predictions[idx][check_labels[idx]])
        last_token_labels.append(labels[idx][check_labels[idx]])

    f1 = f1_metric.compute(predictions=last_token_predictions, references=last_token_labels, average='weighted')["f1"]
    accuracy = accuracy_metric.compute(predictions=last_token_predictions, references=last_token_labels)["accuracy"]
    precision = precision_metric.compute(predictions=last_token_predictions, references=last_token_labels, average='micro')['precision']
    recall = recall_metric.compute(predictions=last_token_predictions, references=last_token_labels, average='micro')['recall']

    return {
        "f1-score": f1,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall
    }


class CustomCallback(TrainerCallback):
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.eval_dataset['train_retain'],
                                   metric_key_prefix="eval_train_retrain")
            self._trainer.evaluate(eval_dataset=self._trainer.eval_dataset['train_forget'],
                                   metric_key_prefix="eval_train_forget")
            return control_copy



class CollatorWithPassThrough:
    """
    Wrap a base collator (e.g., TRL's DataCollatorForCompletionOnlyLM) and
    also pass through dataset columns like `is_forget` and `index`.
    """
    def __init__(self, base_collator):
        self.base_collator = base_collator

    def __call__(self, features):
        # First let the base collator build input_ids/labels/attention_mask
        batch = self.base_collator(features)

        # Then preserve metadata fields needed later
        if 'is_forget' in features[0]:
            batch['is_forget'] = torch.tensor([f['is_forget'] for f in features])
        if 'index' in features[0]:
            batch['index'] = torch.tensor([f['index'] for f in features])

        return batch


def get_data_path(dataset):
    if dataset.lower() == "sst2":
        data_path = "karuna-bhaila/Unlearning_SST2v3"
    elif dataset.lower() == 'yelp':
        data_path = "karuna-bhaila/Unlearning_Yelp_Polarity"
    else:
        # define dataset with the following splits:
        # train_retain, train_forget, test_retain, test_forget
        raise NotImplementedError

    return data_path


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    # argmax to get the token ids
    return logits.argmax(dim=-1)

def get_logits_from_base_model(base_model, data_collator, dataset):
    """
    Save ONLY the first answer-token logits for each sample, aligned to the label
    mask produced by DataCollatorForCompletionOnlyLM.
    """
    base_model.eval()
    loader = DataLoader(dataset['train'], collate_fn=data_collator, batch_size=32)
    original_logits = {}
    for batch in tqdm(loader):
        indices = batch.pop('index')  # [B]
        labels = batch['labels']      # [B, T]
        # move to device
        batch = {k: v.to(base_model.device) if torch.is_tensor(v) else v for k, v in batch.items()}

        with torch.no_grad():
            out = base_model(**{k: batch[k] for k in ('input_ids','attention_mask')})
            logits = out.logits  # [B, T, V]
        # shift to align next-token prediction
        shift_logits = logits[:, :-1, :]     # [B, T-1, V]
        shift_labels = labels[:, 1:]         # [B, T-1]

        # for each row, pick the FIRST position where label != -100
        for b in range(shift_labels.size(0)):
            row = shift_labels[b]  # [T-1]
            idxs = torch.nonzero(row != -100, as_tuple=False).flatten()
            if idxs.numel() == 0:
                continue
            first_pos = idxs[0].item()
            vec = shift_logits[b, first_pos, :].detach().cpu()  # [V]
            key = int(indices[b].item()) if hasattr(indices[b], "item") else int(indices[b])
            original_logits[key] = vec.numpy()
    return original_logits
