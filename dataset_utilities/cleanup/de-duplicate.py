#!/usr/bin/env python3
import csv
import argparse
from pathlib import Path


def deduplicate(csv_path, out_path=None):
    """
    Remove duplicate rows (identical across all columns).
    Keeps the first occurrence, discards subsequent ones.
    """
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        return

    header = rows[0]
    data = rows[1:]

    seen = set()
    result = []

    for row in data:
        key = tuple(row)
        if key not in seen:
            seen.add(key)
            result.append(row)

    if out_path is None:
        out_path = csv_path.with_name(csv_path.stem + "_dedup.csv")

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(result)

    print(f"{csv_path} -> {out_path} (kept {len(result)} rows, removed {len(data)-len(result)} duplicates)")


def main():
    parser = argparse.ArgumentParser(
        description="De-duplicate clone dataset CSV files."
    )
    parser.add_argument(
        "csv_files",
        nargs="+",
        type=Path,
        help="Input CSV files to de-duplicate",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory (otherwise *_dedup.csv next to input)",
    )
    args = parser.parse_args()

    for csv_path in args.csv_files:
        if args.output_dir:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            out_path = args.output_dir / (csv_path.stem + "_dedup.csv")
        else:
            out_path = None
        deduplicate(csv_path, out_path)


if __name__ == "__main__":
    main()
