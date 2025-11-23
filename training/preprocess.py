import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Tuple

from .config import NormalizerConfig


class CodePreprocessor:
    """Applies the configured Java normalizer to both snippets in a pair."""

    def __init__(self, config: NormalizerConfig):
        self.config = config
        self._normalizer = self._load_normalizer() if config.enabled else None

    def _load_normalizer(self) -> ModuleType:
        module_name = self.config.module_name
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError:
            module_path = Path(self.config.module_path)
            if not module_path.is_absolute():
                module_path = Path(__file__).resolve().parent.parent / module_path
            if not module_path.exists():
                raise ImportError(f"Java normalizer module not found at {module_path}") from None
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Unable to load normalizer spec from {module_path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return module

    def _normalize(self, code: str) -> str:
        if not code or not self._normalizer:
            return code
        try:
            normalize = getattr(self._normalizer, "normalize_java_source")
        except AttributeError as exc:
            raise RuntimeError("normalize_java_source function missing in normalizer module") from exc
        try:
            return normalize(
                code,
                strip_comments=self.config.strip_comments,
                mask_strings=self.config.mask_strings,
                mask_numbers=self.config.mask_numbers,
                scope_type_rename=self.config.scope_type_rename,
            )
        except Exception:
            # preserve original snippet if normalizer fails (e.g., optional deps missing)
            return code

    def process_pair(self, left: str, right: str) -> Tuple[str, str]:
        if not self.config.enabled or not self._normalizer:
            return left, right
        return self._normalize(left), self._normalize(right)
