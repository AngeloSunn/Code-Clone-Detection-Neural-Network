import os
import shutil
import logging
import warnings
import random
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import matplotlib.pyplot as plt

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
)
from transformers.trainer_utils import EvalPrediction

# Silence HF warnings if desired
logging.disable(logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------
# Utilities
# ---------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def detect_bf16() -> bool:
    """Use bf16 if available on Ampere+ GPUs."""
    if not torch.cuda.is_available():
        return False
    try:
        major, _ = torch.cuda.get_device_capability()
        return major >= 8  # Ampere or newer
    except Exception:
        return False

# ---------------------------
# Dataset
# ---------------------------

class CloneDetectionDataset(Dataset):
    """
    Expects CSV rows like:
    train:
      col0=folder1, col1=file1, col2=start1, col3=end1,
      col4=folder2, col5=file2, col6=start2, col7=end2, col8=label
    eval (has_label_in_first_col=True):
      col0='<label><folder1>', col1=file1, col2=start1, col3=end1,
      col4=folder2, col5=file2, col6=start2, col7=end2
    """

    def __init__(
        self,
        csv_path: str,
        base_path: str,
        tokenizer,
        max_length: int = 512,
        max_rows: Optional[int] = None,
        has_label_in_first_col: bool = False,
        train_label_col: int = 8,
        cache_files_in_memory: bool = True,
    ):
        self.data = pd.read_csv(csv_path, header=None)
        if max_rows is not None:
            self.data = self.data.iloc[:max_rows].reset_index(drop=True)
        self.base_path = base_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.has_label_in_first_col = has_label_in_first_col
        self.train_label_col = train_label_col
        self.cache_files_in_memory = cache_files_in_memory
        self._file_cache: Dict[str, Any] = {}  # path -> List[str]

    def __len__(self):
        return len(self.data)

    def _read_lines(self, path: str):
        if self.cache_files_in_memory and path in self._file_cache:
            return self._file_cache[path]
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            if self.cache_files_in_memory:
                self._file_cache[path] = lines
            return lines
        except FileNotFoundError:
            print(f"File not found: {path}")
            return []

    def extract_code(self, folder, filename, start, end):
        path = os.path.join(self.base_path, str(folder), str(filename))
        lines = self._read_lines(path)
        try:
            s = int(start)
            e = int(end)
        except Exception:
            s, e = 0, 0
        snippet = lines[s:e] if 0 <= s <= e <= len(lines) else []
        return "".join(snippet)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        if self.has_label_in_first_col:
            # first char of col0 is label, remainder is folder name
            label = int(str(row[0])[0])
            folder1 = str(row[0])[1:]
            file1, start1, end1 = row[1], row[2], row[3]
            folder2, file2, start2, end2 = row[4], row[5], row[6], row[7]
        else:
            label = int(row[self.train_label_col])
            folder1, file1, start1, end1 = row[0], row[1], row[2], row[3]
            folder2, file2, start2, end2 = row[4], row[5], row[6], row[7]

        code1 = self.extract_code(folder1, file1, start1, end1)
        code2 = self.extract_code(folder2, file2, start2, end2)

        # Return plain Python types; DataCollator will pad and convert to tensors
        tokens = self.tokenizer(
            code1,
            code2,
            truncation=True,
            max_length=self.max_length,
        )
        tokens["labels"] = label
        return tokens

# ---------------------------
# Metrics
# ---------------------------

def compute_metrics(eval_pred: EvalPrediction):
    logits = eval_pred.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    labels = eval_pred.label_ids
    preds = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="binary", zero_division=0),
        "precision": precision_score(labels, preds, average="binary", zero_division=0),
        "recall": recall_score(labels, preds, average="binary", zero_division=0),
    }

# ---------------------------
# Plotting
# ---------------------------

def plot_metrics(trainer: Trainer):
    logs = trainer.state.log_history

    # Persist raw logs to CSV
    os.makedirs("logs", exist_ok=True)
    df_logs = pd.DataFrame(logs)
    df_logs.to_csv("logs/trainer_log_history.csv", index=False)

    # Extract series
    train_loss = [log["loss"] for log in logs if "loss" in log]
    eval_loss = [log["eval_loss"] for log in logs if "eval_loss" in log]
    eval_acc = [log["eval_accuracy"] for log in logs if "eval_accuracy" in log]
    eval_f1 = [log["eval_f1"] for log in logs if "eval_f1" in log]
    learning_rate = [log["learning_rate"] for log in logs if "learning_rate" in log]

    train_steps = [i for i, log in enumerate(logs) if "loss" in log]
    eval_steps = [i for i, log in enumerate(logs) if "eval_accuracy" in log]
    lr_steps = [i for i, log in enumerate(logs) if "learning_rate" in log]

    def _save_plot(xs, ys, title, xlabel, ylabel, filename):
        if len(xs) == 0 or len(ys) == 0:
            return
        plt.figure()
        plt.plot(xs, ys, label=title)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    _save_plot(train_steps, train_loss, "Training Loss", "Training Step", "Loss", "plot_train_loss.png")
    _save_plot(eval_steps, eval_acc, "Evaluation Accuracy", "Eval Step", "Accuracy", "plot_eval_accuracy.png")
    _save_plot(eval_steps, eval_loss, "Evaluation Loss", "Eval Step", "Loss", "plot_eval_loss.png")
    _save_plot(eval_steps, eval_f1, "F1 Score", "Eval Step", "F1", "plot_f1_score.png")
    _save_plot(lr_steps, learning_rate, "Learning Rate", "Step", "LR", "plot_learning_rate.png")

# ---------------------------
# Main
# ---------------------------

def main():
    set_seed(42)
    torch.backends.cudnn.benchmark = True

    print("CUDA available:", torch.cuda.is_available())
    print("bf16 available:", detect_bf16())

    output_dir = "./codebert_clone_model_full_250-warmup"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print("removing previous runs")

    model_name = "microsoft/codebert-base"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name, num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

    train_dataset = CloneDetectionDataset(
        csv_path="training_data_full/train.csv",
        base_path="BigCloneBench/dataset",
        tokenizer=tokenizer,
        max_length=512,
        #max_rows=500000,
        has_label_in_first_col=False,  # train CSV must include label in col 8
        train_label_col=8,
        cache_files_in_memory=True,
    )

    eval_dataset = CloneDetectionDataset(
        csv_path="training_data_full/eval.csv",
        base_path="BigCloneBench/dataset",
        tokenizer=tokenizer,
        max_length=512,
        #max_rows=25000,
        has_label_in_first_col=True,   # label encoded in col0 first char
        cache_files_in_memory=True,
    )

    # Data collator pads to the longest in the batch and aligns to multiples of 8 for Tensor Cores
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

    # Training hyperparameters
    use_bf16 = detect_bf16()
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,  # consider driving with max_steps for very large sets
        per_device_train_batch_size=96,
        per_device_eval_batch_size=96,
        gradient_accumulation_steps=2,
        learning_rate=2e-5, # war 2e-5 für die ersten beiden
        weight_decay=0.01,
        warmup_steps=500,
        #warmup_ratio=0.06, # war 0.6 für die ersten beiden
        lr_scheduler_type="cosine",

        evaluation_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,

        logging_dir="./logs",
        logging_steps=100,
        report_to=["tensorboard"],

        fp16=(not use_bf16),
        bf16=use_bf16,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        gradient_checkpointing=False,  # set True if VRAM is tight
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    # Train
    trainer.train()

    # Save best
    best_dir = os.path.join(output_dir, "best_model")
    os.makedirs(best_dir, exist_ok=True)
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)

    # Plots + logs
    plot_metrics(trainer)
    print("Training complete. Artifacts:")
    print(f"- Model dir: {best_dir}")
    print("- Plots: plot_train_loss.png, plot_eval_accuracy.png, plot_eval_loss.png, plot_f1_score.png, plot_learning_rate.png")
    print("- Logs CSV: logs/trainer_log_history.csv")

if __name__ == "__main__":
    main()
