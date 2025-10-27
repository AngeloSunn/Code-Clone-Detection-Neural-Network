from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataConfig:
    train_csv: str = "datasets/training_data_full/train.csv"
    eval_csv: str = "datasets/training_data_full/eval.csv"
    test_csv: Optional[str] = None
    label_col: int = 8
    base_path: str = "BigCloneBench/dataset"
    max_length: int = 512
    max_rows: Optional[int] = None
    cache_files_in_memory: bool = True
    apply_preprocessing: bool = False


@dataclass
class ModelConfig:
    pretrained_model_name: str = "microsoft/codebert-base"
    num_labels: int = 2


@dataclass
class NormalizerConfig:
    enabled: bool = False
    module_name: str = "java_normalizer"
    module_path: str = "java normalizer/java_normalizer.py"
    strip_comments: bool = True
    mask_strings: bool = True
    mask_numbers: bool = True
    scope_type_rename: bool = True


@dataclass
class TrainerConfig:
    output_dir: str = "trained_models/test"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 6
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    lr_scheduler_type: str = "cosine"
    eval_strategy: str = "steps"
    eval_steps: int = 100
    save_strategy: str = "steps"
    save_steps: int = 100
    save_total_limit: int = 1
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "accuracy"
    greater_is_better: bool = True
    logging_dir: str = "./logs"
    logging_steps: int = 100
    report_to_tensorboard: bool = True
    dataloader_num_workers: int = 4
    gradient_checkpointing: bool = False
    early_stopping_patience: int = 5


@dataclass
class ExperimentConfig:
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    normalizer: NormalizerConfig = field(default_factory=NormalizerConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
