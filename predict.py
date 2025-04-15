from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
import sys

# Load fine-tuned model
model_path = "./codebert_clone_model"
model = RobertaForSequenceClassification.from_pretrained(model_path)
tokenizer = RobertaTokenizer.from_pretrained(model_path)

# Your input code segments
code1 = """def add(a, b): return a + b"""
code2 = """def mult(x, y): return x * y"""

# Tokenize
inputs = tokenizer(code1, code2, return_tensors="pt", truncation=True, padding="max_length", max_length=512)

# Predict
with torch.no_grad():
    outputs = model(**inputs)
    clone_score = outputs.logits.squeeze().item()
    clone_percentage = max(0, min(1, clone_score)) * 100

print(f"Clone Likelihood: {clone_percentage:.2f}%")
