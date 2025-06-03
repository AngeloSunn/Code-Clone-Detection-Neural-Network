import torch
from torch.utils.data import Dataset
from transformers import (
    RobertaTokenizer,
    RobertaConfig,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
import pandas as pd
import os
import warnings
import matplotlib.pyplot as plt
import logging
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

logging.disable(logging.WARNING)

class CloneDetectionDataset(Dataset):
    def __init__(self, csv_path, base_path, tokenizer, max_length=512, max_rows=None, has_label_in_first_col=False):
        self.data = pd.read_csv(csv_path, header=None)
        if max_rows is not None:
            self.data = self.data.iloc[:max_rows]
        self.base_path = base_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.has_label_in_first_col = has_label_in_first_col

    def __len__(self):
        return len(self.data)

    def extract_code(self, folder, filename, start, end):
        path = os.path.join(self.base_path, folder, filename)
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
                snippet = lines[int(start):int(end)]
                return ''.join(snippet)
        except FileNotFoundError:
            print(f"⚠️ File not found: {path}")
            return ""

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        if self.has_label_in_first_col:
            # Extract label from first character of first column
            label = int(str(row[0])[0])  # '1' or '0'
            folder1 = str(row[0])[1:]    # folder name after first char
            file1, start1, end1 = row[1], row[2], row[3]
            folder2, file2, start2, end2 = row[4], row[5], row[6], row[7]
        else:
            # Training data: label by parity of idx (adjust if needed)
            label = 1 if idx % 2 == 0 else 0
            folder1, file1, start1, end1 = row[0], row[1], row[2], row[3]
            folder2, file2, start2, end2 = row[4], row[5], row[6], row[7]

        code1 = self.extract_code(folder1, file1, start1, end1)
        code2 = self.extract_code(folder2, file2, start2, end2)

        tokens = self.tokenizer(
            code1,
            code2,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
            return_overflowing_tokens=False
        )
        tokens = {k: v.squeeze(0) for k, v in tokens.items()}
        # Use long dtype for classification labels
        tokens["labels"] = torch.tensor(label, dtype=torch.long)
        return tokens

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }

def plot_metrics(trainer):
    logs = trainer.state.log_history

    # Extract metrics
    train_loss = [log["loss"] for log in logs if "loss" in log]
    eval_loss = [log["eval_loss"] for log in logs if "eval_loss" in log]
    eval_acc = [log["eval_accuracy"] for log in logs if "eval_accuracy" in log]
    f1 = [log["eval_f1"] for log in logs if "eval_f1" in log]
    learning_rate = [log["learning_rate"] for log in logs if "learning_rate" in log]

    # x-axis indexes
    train_steps = [i for i, log in enumerate(logs) if "loss" in log]
    eval_steps = [i for i, log in enumerate(logs) if "eval_accuracy" in log]
    lr_steps = [i for i, log in enumerate(logs) if "learning_rate" in log]

    # Plot 1: Training Loss
    plt.figure()
    plt.plot(train_steps, train_loss, label="Train Loss")
    plt.title("Training Loss")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("plot_train_loss.png")
    plt.close()

    # Plot 2: Evaluation Accuracy
    plt.figure()
    plt.plot(eval_steps, eval_acc, label="Eval Accuracy", color="green")
    plt.title("Evaluation Accuracy")
    plt.xlabel("Eval Step")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("plot_eval_accuracy.png")
    plt.close()

    # Plot 3: Evaluation Loss
    plt.figure()
    plt.plot(eval_steps, eval_loss, label="Eval Loss", color="orange")
    plt.title("Evaluation Loss")
    plt.xlabel("Eval Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("plot_eval_loss.png")
    plt.close()

    # Plot 4: F1 Score
    plt.figure()
    plt.plot(eval_steps, f1, label="F1 Score", color="blue")
    plt.title("F1 Score")
    plt.xlabel("Eval Step")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.savefig("plot_f1_score.png")
    plt.close()

    # Plot 5: Learning Rate
    plt.figure()
    plt.plot(lr_steps, learning_rate, label="Learning Rate", color="purple")
    plt.title("Learning Rate")
    plt.xlabel("Step")
    plt.ylabel("LR")
    plt.legend()
    plt.savefig("plot_learning_rate.png")
    plt.close()

def main():
    print("CUDA available:", torch.cuda.is_available())

    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    config = RobertaConfig.from_pretrained("microsoft/codebert-base", num_labels=2)
    model = RobertaForSequenceClassification.from_pretrained("microsoft/codebert-base", config=config)

    train_dataset = CloneDetectionDataset(
        csv_path="Data9010CDLH/LearnData.csv",
        base_path="BigCloneBench/dataset",
        tokenizer=tokenizer,
        max_rows=5000,
        has_label_in_first_col=False
    )

    eval_dataset = CloneDetectionDataset(
        csv_path="Data9010CDLH/EvaluationData.csv",
        base_path="BigCloneBench/dataset",
        tokenizer=tokenizer,
        max_rows=2500,
        has_label_in_first_col=True
    )

    training_args = TrainingArguments(
        output_dir="./codebert_clone_model",
        num_train_epochs=10,
        per_device_train_batch_size=24,
        per_device_eval_batch_size=24,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()
    trainer.save_model("./codebert_clone_model/best_model")
    tokenizer.save_pretrained("./codebert_clone_model/best_model")

    plot_metrics(trainer)

if __name__ == "__main__":
    main()
