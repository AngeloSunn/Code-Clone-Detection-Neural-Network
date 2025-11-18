#!/usr/bin/env python3
import csv
import argparse
from pathlib import Path


def measure_balance(csv_path):
    """
    Count how many rows have Class=1 (clone) and Class=0 (non-clone).
    Assumes Class is the last column.
    """
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        return 0, 0

    data = rows[1:]  # skip header
    n_clone = 0
    n_non = 0

    for row in data:
        if not row:
            continue
        cls = row[-1]
        if cls == "1":
            n_clone += 1
        elif cls == "0":
            n_non += 1

    return n_clone, n_non


def main():
    parser = argparse.ArgumentParser(
        description="Measure class balance (clone vs non-clone) in CSV file(s)."
    )
    parser.add_argument(
        "csv_files", nargs="+", type=Path, help="CSV file(s) to analyze"
    )
    args = parser.parse_args()

    for csv_path in args.csv_files:
        n_clone, n_non = measure_balance(csv_path)
        total = n_clone + n_non
        print(f"File: {csv_path}")
        print(f"  total:      {total}")
        print(f"  clones:     {n_clone}")
        print(f"  non-clones: {n_non}")
        if total > 0:
            print(f"  clone ratio: {n_clone/total:.3f}")
        print()


if __name__ == "__main__":
    main()
