#!/usr/bin/env python3
import csv
import argparse
import random
from pathlib import Path


def rebalance(csv_path, seed=None, out_path=None):
    """
    Down-sample the majority class so that the dataset has
    equal numbers of Class=1 and Class=0 rows.
    """
    if seed is not None:
        random.seed(seed)

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        return

    header = rows[0]
    data = rows[1:]

    clones = []
    non_clones = []

    for row in data:
        if len(row) == 0:
            continue
        cls = row[-1]
        if cls == "1":
            clones.append(row)
        elif cls == "0":
            non_clones.append(row)

    k = min(len(clones), len(non_clones))
    if k == 0:
        print(f"{csv_path}: cannot rebalance (one class is empty).")
        return

    clones_sample = random.sample(clones, k)
    non_sample = random.sample(non_clones, k)

    balanced = clones_sample + non_sample
    random.shuffle(balanced)

    if out_path is None:
        out_path = csv_path.with_name(csv_path.stem + "_balanced.csv")

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(balanced)

    print(
        f"{csv_path} -> {out_path} "
        f"(clones={len(clones)}, non-clones={len(non_clones)}, kept per class={k})"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Re-balance clone dataset CSV files by down-sampling the majority class."
    )
    parser.add_argument(
        "csv_files",
        nargs="+",
        type=Path,
        help="Input CSV files to re-balance",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory (otherwise *_balanced.csv next to input)",
    )
    args = parser.parse_args()

    for csv_path in args.csv_files:
        if args.output_dir:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            out_path = args.output_dir / (csv_path.stem + "_balanced.csv")
        else:
            out_path = None
        rebalance(csv_path, seed=args.seed, out_path=out_path)


if __name__ == "__main__":
    main()
