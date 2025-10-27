import os
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


def plot_metrics(trainer, output_dir: str = "logs"):
    logs = trainer.state.log_history
    os.makedirs(output_dir, exist_ok=True)
    df_logs = pd.DataFrame(logs)
    csv_path = os.path.join(output_dir, "trainer_log_history.csv")
    df_logs.to_csv(csv_path, index=False)

    train_loss = [log["loss"] for log in logs if "loss" in log]
    eval_loss = [log["eval_loss"] for log in logs if "eval_loss" in log]
    eval_acc = [log["eval_accuracy"] for log in logs if "eval_accuracy" in log]
    eval_f1 = [log["eval_f1"] for log in logs if "eval_f1" in log]
    learning_rate = [log["learning_rate"] for log in logs if "learning_rate" in log]

    train_steps = [i for i, log in enumerate(logs) if "loss" in log]
    eval_steps = [i for i, log in enumerate(logs) if "eval_accuracy" in log]
    lr_steps = [i for i, log in enumerate(logs) if "learning_rate" in log]

    def _save_plot(xs, ys, title, xlabel, ylabel, filename):
        if not xs or not ys:
            return
        plt.figure()
        plt.plot(xs, ys, label=title)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()

    _save_plot(train_steps, train_loss, "Training Loss", "Training Step", "Loss", "plot_train_loss.png")
    _save_plot(eval_steps, eval_acc, "Evaluation Accuracy", "Eval Step", "Accuracy", "plot_eval_accuracy.png")
    _save_plot(eval_steps, eval_loss, "Evaluation Loss", "Eval Step", "Loss", "plot_eval_loss.png")
    _save_plot(eval_steps, eval_f1, "F1 Score", "Eval Step", "F1", "plot_f1_score.png")
    _save_plot(lr_steps, learning_rate, "Learning Rate", "Step", "LR", "plot_learning_rate.png")

    plot_files = [
        "plot_train_loss.png",
        "plot_eval_accuracy.png",
        "plot_eval_loss.png",
        "plot_f1_score.png",
        "plot_learning_rate.png",
    ]

    return {
        "logs_csv": csv_path,
        "plots": [os.path.join(output_dir, name) for name in plot_files],
    }
