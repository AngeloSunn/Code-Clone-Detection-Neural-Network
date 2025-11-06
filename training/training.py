import os
import shutil
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root))

import torch
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
)

from training.config import ExperimentConfig
from training.dataset import CloneDetectionDataset
from training.fetcher import CodeSnippetFetcher
from training.metrics import compute_metrics
from training.plotting import plot_metrics
from training.preprocess import CodePreprocessor
from training.utils import (
    detect_bf16,
    set_seed,
    silence_transformers_warnings,
)

from typing import Optional

def prepare_output_dir(path: str):
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"Removed previous training artifacts at {path}")
    os.makedirs(path, exist_ok=True)


def build_datasets(cfg: ExperimentConfig, tokenizer):
    fetcher = CodeSnippetFetcher(
        base_path=cfg.data.base_path,
        cache_files_in_memory=cfg.data.cache_files_in_memory,
    )
    preprocessor = None
    if cfg.normalizer.enabled:
        try:
            preprocessor = CodePreprocessor(cfg.normalizer)
        except ImportError as exc:
            print(f"Warning: failed to load normalizer module: {exc}")

    train_dataset = CloneDetectionDataset(
        csv_path=cfg.data.train_csv,
        tokenizer=tokenizer,
        fetcher=fetcher,
        max_length=cfg.data.max_length,
        max_rows=cfg.data.max_rows,
        label_col=cfg.data.label_col,
        preprocessor=preprocessor,
        apply_preprocessing=cfg.data.apply_preprocessing or cfg.normalizer.enabled,
    )

    eval_dataset = CloneDetectionDataset(
        csv_path=cfg.data.eval_csv,
        tokenizer=tokenizer,
        fetcher=fetcher,
        max_length=cfg.data.max_length,
        max_rows=cfg.data.max_rows,
        label_col=cfg.data.label_col,
        preprocessor=preprocessor,
        apply_preprocessing=cfg.data.apply_preprocessing or cfg.normalizer.enabled,
    )
    
    test_dataset = None
    if cfg.data.test_csv:
        test_dataset = CloneDetectionDataset(
            csv_path=cfg.data.test_csv,
            tokenizer=tokenizer,
            fetcher=fetcher,
            max_length=cfg.data.max_length,
            max_rows=cfg.data.max_rows,
            label_col=cfg.data.label_col,
            preprocessor=preprocessor,
            apply_preprocessing=cfg.data.apply_preprocessing or cfg.normalizer.enabled,
        )
    return train_dataset, eval_dataset, test_dataset


def build_trainer(cfg: ExperimentConfig):
    silence_transformers_warnings()
    set_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True

    use_bf16 = detect_bf16()

    print("CUDA available:", torch.cuda.is_available())
    print("bf16 available:", use_bf16)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.pretrained_model_name)
    model_config = AutoConfig.from_pretrained(
        cfg.model.pretrained_model_name,
        num_labels=cfg.model.num_labels,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.pretrained_model_name,
        config=model_config,
    )

    train_dataset, eval_dataset, test_dataset = build_datasets(cfg, tokenizer)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

    report_to = ["tensorboard"] if cfg.trainer.report_to_tensorboard else []

    training_args = TrainingArguments(
        output_dir=cfg.trainer.output_dir,
        num_train_epochs=cfg.trainer.num_train_epochs,
        per_device_train_batch_size=cfg.trainer.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.trainer.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.trainer.gradient_accumulation_steps,
        learning_rate=cfg.trainer.learning_rate,
        weight_decay=cfg.trainer.weight_decay,
        warmup_steps=cfg.trainer.warmup_steps,
        lr_scheduler_type=cfg.trainer.lr_scheduler_type,
        eval_strategy=cfg.trainer.eval_strategy,
        eval_steps=cfg.trainer.eval_steps,
        save_strategy=cfg.trainer.save_strategy,
        save_steps=cfg.trainer.save_steps,
        save_total_limit=cfg.trainer.save_total_limit,
        load_best_model_at_end=cfg.trainer.load_best_model_at_end,
        metric_for_best_model=cfg.trainer.metric_for_best_model,
        greater_is_better=cfg.trainer.greater_is_better,
        logging_dir=cfg.trainer.logging_dir,
        logging_steps=cfg.trainer.logging_steps,
        report_to=report_to,
        fp16=(not use_bf16),
        bf16=use_bf16,
        dataloader_num_workers=cfg.trainer.dataloader_num_workers,
        remove_unused_columns=False,
        gradient_checkpointing=cfg.trainer.gradient_checkpointing,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.trainer.early_stopping_patience)],
    )
    return trainer, tokenizer, test_dataset


def main(cfg: Optional[ExperimentConfig] = None):
    cfg = cfg or ExperimentConfig()
    prepare_output_dir(cfg.trainer.output_dir)

    trainer, tokenizer, test_dataset = build_trainer(cfg)

    trainer.train()

    best_dir = os.path.join(cfg.trainer.output_dir, "best_model")
    os.makedirs(best_dir, exist_ok=True)
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)

    artifacts = plot_metrics(trainer, output_dir=cfg.trainer.logging_dir)
    print("Training complete. Artifacts:")
    print(f"- Model dir: {best_dir}")
    print(f"- Logs CSV: {artifacts['logs_csv']}")
    print(f"- Plots: {', '.join(artifacts['plots'])}")

    if test_dataset is not None:
        print(f"Running final evaluation on test dataset: {cfg.data.test_csv}")
        test_metrics = trainer.evaluate(test_dataset, metric_key_prefix="test")
        trainer.log_metrics("test", test_metrics)
        trainer.save_metrics("test", test_metrics)
        print("Test metrics:")
        for key, value in sorted(test_metrics.items()):
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
