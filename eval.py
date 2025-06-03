import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import pandas as pd
import os
from tqdm import tqdm
import logging

logging.disable(logging.WARNING)
# === CONFIG ===
MODEL_PATH = "./codebert_clone_model/best_model"
CSV_PATH = "Data9010CDLH/EvaluationData.csv"
BASE_PATH = "BigCloneBench/dataset"
BATCH_SIZE = 100
MAX_LENGTH = 512
MAX_ROWS = 2500  # <-- Evaluate only first 2500 lines

# === Dataset class ===
class CloneDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, base_path, tokenizer, max_length=512, max_rows=None):
        self.data = pd.read_csv(csv_path, header=None)
        if max_rows:
            self.data = self.data.iloc[:max_rows]
        self.base_path = base_path
        self.tokenizer = tokenizer
        self.max_length = max_length

    def extract_code(self, folder, filename, start, end):
        path = os.path.join(self.base_path, folder, filename)
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
                return ''.join(lines[int(start):int(end)])
        except FileNotFoundError:
            return ""

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        label = int(str(row[0])[0])  # label is first char of first column
        folder1, file1, start1, end1 = str(row[0])[1:], row[1], row[2], row[3]
        folder2, file2, start2, end2 = row[4], row[5], row[6], row[7]

        code1 = self.extract_code(folder1, file1, start1, end1)
        code2 = self.extract_code(folder2, file2, start2, end2)

        tokens = self.tokenizer(
            code1,
            code2,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }

# === Evaluation logic ===
def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)
    model.to(device)
    model.eval()

    dataset = CloneDetectionDataset(CSV_PATH, BASE_PATH, tokenizer, MAX_LENGTH, max_rows=MAX_ROWS)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

    correct = 0
    total = 0
    count_positive = 0
    count_negative = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predicted_labels = torch.argmax(logits, dim=1)

            correct += (predicted_labels == labels).sum().item()
            total += labels.size(0)

            count_positive += (predicted_labels == 1).sum().item()
            count_negative += (predicted_labels == 0).sum().item()

    print(f"✅ Accuracy on first {MAX_ROWS} examples: {correct}/{total} = {correct / total:.4f}")
    print(f"Model predicted Positive: {count_positive} times")
    print(f"Model predicted Negative: {count_negative} times")


if __name__ == "__main__":
    evaluate()
