#!/usr/bin/env python3
import csv
import argparse
from collections import Counter
from pathlib import Path


def detect_duplicates(csv_path):
    """
    Return list of duplicate row tuples (excluding header).
    A duplicate is any full row appearing more than once.
    """
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        return [], []

    header = rows[0]
    data = rows[1:]

    counter = Counter(tuple(row) for row in data)
    duplicates = [row for row, cnt in counter.items() if cnt > 1]
    return header, duplicates


def main():
    parser = argparse.ArgumentParser(
        description="Detect duplicate rows in a single CSV file."
    )
    parser.add_argument("csv_file", type=Path, help="Path to input CSV file")
    parser.add_argument(
        "--show-rows",
        action="store_true",
        help="Print duplicate rows themselves",
    )
    args = parser.parse_args()

    header, duplicates = detect_duplicates(args.csv_file)
    print(f"File: {args.csv_file}")
    print(f"Number of distinct duplicate rows: {len(duplicates)}")

    if args.show_rows and duplicates:
        print("Header:", header)
        for row in duplicates:
            print(row)


if __name__ == "__main__":
    main()
