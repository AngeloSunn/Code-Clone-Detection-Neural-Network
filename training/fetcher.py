import os
from typing import Dict, List


class CodeSnippetFetcher:
    """Loads code snippets from disk with optional in-memory caching."""

    def __init__(self, base_path: str, cache_files_in_memory: bool = True):
        self.base_path = base_path
        self.cache_files_in_memory = cache_files_in_memory
        self._file_cache: Dict[str, List[str]] = {}

    def _read_lines(self, path: str) -> List[str]:
        if self.cache_files_in_memory and path in self._file_cache:
            return self._file_cache[path]
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as file:
                lines = file.readlines()
        except FileNotFoundError:
            # mirror old behaviour: return empty snippet when file is missing.
            return []
        if self.cache_files_in_memory:
            self._file_cache[path] = lines
        return lines

    def fetch(self, folder: str, filename: str, start: int, end: int) -> str:
        path = os.path.join(self.base_path, str(folder), str(filename))
        lines = self._read_lines(path)
        try:
            begin = int(start)
            finish = int(end)
        except (TypeError, ValueError):
            begin, finish = 0, 0

        if begin < 0 or finish < begin or finish > len(lines):
            return ""
        snippet = lines[begin:finish]
        return "".join(snippet)
