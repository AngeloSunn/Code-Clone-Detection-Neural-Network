from typing import Optional, Tuple

import pandas as pd
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from .fetcher import CodeSnippetFetcher
from .preprocess import CodePreprocessor


class CloneDetectionDataset(Dataset):
    """
    Dataset that loads clone detection pairs from CSV metadata using a fetcher.
    """

    def __init__(
        self,
        csv_path: str,
        tokenizer: PreTrainedTokenizerBase,
        fetcher: CodeSnippetFetcher,
        max_length: int = 512,
        max_rows: Optional[int] = None,
        label_col: int = 8,
        preprocessor: Optional[CodePreprocessor] = None,
        apply_preprocessing: bool = False,
    ):
        self.data = pd.read_csv(csv_path, header=None)
        if max_rows is not None:
            self.data = self.data.iloc[:max_rows].reset_index(drop=True)

        self.tokenizer = tokenizer
        self.fetcher = fetcher
        self.max_length = max_length
        self.label_col = label_col
        self.preprocessor = preprocessor
        self.apply_preprocessing = apply_preprocessing and preprocessor is not None

    def __len__(self) -> int:
        return len(self.data)

    def _extract_code(self, folder, filename, start, end) -> str:
        return self.fetcher.fetch(folder, filename, start, end)

    def _apply_preprocessing(self, code1: str, code2: str) -> Tuple[str, str]:
        if self.apply_preprocessing and self.preprocessor:
            return self.preprocessor.process_pair(code1, code2)
        return code1, code2

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]

        label = int(row[self.label_col])
        folder1, file1, start1, end1 = row[0], row[1], row[2], row[3]
        folder2, file2, start2, end2 = row[4], row[5], row[6], row[7]

        code1 = self._extract_code(folder1, file1, start1, end1)
        code2 = self._extract_code(folder2, file2, start2, end2)

        code1, code2 = self._apply_preprocessing(code1, code2)

        tokens = self.tokenizer(
            code1,
            code2,
            truncation=True,
            max_length=self.max_length,
        )
        tokens["labels"] = label
        return tokens
