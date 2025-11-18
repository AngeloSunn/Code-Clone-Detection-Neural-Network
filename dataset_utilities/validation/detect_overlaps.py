#!/usr/bin/env python3
import csv
import argparse
from pathlib import Path
from collections import defaultdict


def extract_function_left(row):
    # (Folder1, File1, BeginLine1, EndLine1) -> columns 0..3
    return tuple(row[0:4])


def extract_function_right(row):
    # (Folder2, File2, BeginLine2, EndLine2) -> columns 4..8 (but we only need 4..7)
    return tuple(row[4:8])


def detect_overlaps(csv_paths):
    """
    For each CSV file, build a set of functions it contains (left and right).
    Then return all functions that occur in more than one file with the list
    of files in which they appear.
    """
    func_to_files = defaultdict(set)

    for csv_path in csv_paths:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)

        if not rows:
            continue

        data = rows[1:]  # skip header
        functions_in_file = set()

        for row in data:
            if len(row) < 9:
                continue  # malformed row, skip
            f1 = extract_function_left(row)
            f2 = extract_function_right(row)
            functions_in_file.add(f1)
            functions_in_file.add(f2)

        for func in functions_in_file:
            func_to_files[func].add(str(csv_path))

    # Functions that appear in > 1 file
    overlaps = {func: files for func, files in func_to_files.items() if len(files) > 1}
    return overlaps


def main():
    parser = argparse.ArgumentParser(
        description="Detect functions that appear in more than one CSV file."
    )
    parser.add_argument(
        "csv_files",
        nargs="+",
        type=Path,
        help="CSV files to check for function overlaps",
    )
    args = parser.parse_args()

    overlaps = detect_overlaps(args.csv_files)
    count = len(overlaps)
    print(f"Number of overlapping functions: {count}")


if __name__ == "__main__":
    main()
