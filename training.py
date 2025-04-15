from transformers import RobertaTokenizer, RobertaModel, RobertaConfig, RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, Value
import torch
from transformers import DataCollatorWithPadding
import pandas as pd

print(torch.cuda.is_available())  # This should return True if your GPU is available
# Load CSV into HuggingFace dataset
dataset = load_dataset("csv", data_files={"train": "train_clone_dataset.csv", "test": "test_clone_dataset.csv"})

# Load tokenizer
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

# Tokenize inputs: code1 and code2 as a pair
def tokenize_function(examples):
    # Tokenize the code pairs (code1 and code2) and add the labels as well
    tokenized = tokenizer(examples["code1"], examples["code2"], truncation=True, padding="max_length")
    tokenized["label"] = examples["clone_label"]  # Add the labels for regression task
    return tokenized

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Convert label to float (for regression)
tokenized_datasets = tokenized_datasets.cast_column("clone_label", Value("float32"))

# Load model (regression head)
config = RobertaConfig.from_pretrained("microsoft/codebert-base", num_labels=1, problem_type="regression")
model = RobertaForSequenceClassification.from_pretrained("microsoft/codebert-base", config=config)

# Training args
training_args = TrainingArguments(
    output_dir="./codebert_clone_model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    load_best_model_at_end=True,
)

# MSE metric
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.squeeze()
    return {"mse": ((predictions - labels) ** 2).mean()}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
)

trainer.train()
model.save_pretrained("./codebert_clone_model")
tokenizer.save_pretrained("./codebert_clone_model")
