import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers.trainer_utils import EvalPrediction


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
