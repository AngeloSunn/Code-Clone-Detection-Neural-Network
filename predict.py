import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import argparse
import torch.nn.functional as F

def load_model(model_path):
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    model = RobertaForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model

def predict_classification(code1, code2, tokenizer, model, max_length=512):
    inputs = tokenizer(
        code1,
        code2,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.squeeze()
        probs = F.softmax(logits, dim=-1)

        predicted_class = torch.argmax(probs).item()
        confidence = probs[predicted_class].item()
        label = "CLONE" if predicted_class == 1 else "NOT CLONE"

        return label, confidence

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./codebert_clone_model/best_model", help="Path to trained model")
    parser.add_argument("--file1", type=str, required=True, help="Path to first code file")
    parser.add_argument("--file2", type=str, required=True, help="Path to second code file")
    args = parser.parse_args()

    tokenizer, model = load_model(args.model_path)

    with open(args.file1, "r", encoding="utf-8", errors="ignore") as f:
        code1 = f.read()

    with open(args.file2, "r", encoding="utf-8", errors="ignore") as f:
        code2 = f.read()

    label, confidence = predict_classification(code1, code2, tokenizer, model)
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.4f}")
