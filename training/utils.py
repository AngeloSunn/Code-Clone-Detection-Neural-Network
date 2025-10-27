import logging
import random
import warnings

import numpy as np
import torch


def silence_transformers_warnings():
    logging.disable(logging.WARNING)
    warnings.filterwarnings("ignore", category=UserWarning)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def detect_bf16() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        major, _ = torch.cuda.get_device_capability()
        return major >= 8
    except Exception:
        return False
