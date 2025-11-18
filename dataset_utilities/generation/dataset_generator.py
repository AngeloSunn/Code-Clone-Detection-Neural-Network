#!/usr/bin/env python3
import csv
import random
import argparse
from pathlib import Path


def read_pairs_from_csv(csv_path):
    """Read labeled pairs from a single CSV file."""
    pairs_clone = []
    pairs_nonclone = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)

        for row in reader:
            # last column is class label
            label = row[-1]
            pair = tuple(row[:-1])  # function tuple data without label
            if label == "1":
                pairs_clone.append(pair)
            elif label == "0":
                pairs_nonclone.append(pair)

    return pairs_clone, pairs_nonclone


def read_unlabeled_pairs(csv_path):
    """Read pairs from a CSV file containing only clone or only non-clone pairs."""
    pairs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            pair = tuple(row[:-1])
            pairs.append(pair)
    return pairs


def extract_function_ids(pair):
    """Extract the two function identifiers from a row tuple."""
    # left function: columns 0..3
    f1 = tuple(pair[0:4])
    # right function: columns 4..8 (but only need 4..7)
    f2 = tuple(pair[4:8])
    return f1, f2


def seeded_partition(functions, a, b, c, seed):
    """Seeded partition of functions into train/val/test based on split fractions."""
    random.seed(seed)
    
    S = sorted(functions)  
    random.shuffle(S)

    N = len(S)
    k1 = int(a * N)
    k2 = int(b * N)

    F_train = S[:k1]
    F_val = S[k1:k1 + k2]
    F_test = S[k1 + k2:]

    return F_train, F_val, F_test



def write_csv(pairs, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for p in pairs:
            writer.writerow(p)


def main():
    parser = argparse.ArgumentParser(description="Dataset generator")
    parser.add_argument("--mode", choices=["single", "double"], required=True)
    parser.add_argument("--csv_labeled")
    parser.add_argument("--csv_clone")
    parser.add_argument("--csv_nonclone")
    parser.add_argument("--train", type=float, required=True)
    parser.add_argument("--val", type=float, required=True)
    parser.add_argument("--test", type=float, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # --- load input pairs ---
    clone_pairs = []
    nonclone_pairs = []

    if args.mode == "single":
        if not args.csv_labeled:
            raise ValueError("csv_labeled required in single mode")
        clone_pairs, nonclone_pairs = read_pairs_from_csv(args.csv_labeled)

    else:  # double
        if not (args.csv_clone and args.csv_nonclone):
            raise ValueError("csv_clone and csv_nonclone are required in double mode")
        clone_pairs = read_unlabeled_pairs(args.csv_clone)
        nonclone_pairs = read_unlabeled_pairs(args.csv_nonclone)

    # --- collect all functions ---
    functions = set()
    for cp in clone_pairs:
        f1, f2 = extract_function_ids(cp)
        functions.add(f1)
        functions.add(f2)
    for ncp in nonclone_pairs:
        f1, f2 = extract_function_ids(ncp)
        functions.add(f1)
        functions.add(f2)

    # --- split functions ---
    F_train, F_val, F_test = seeded_partition(
        list(functions),
        args.train, args.val, args.test,
        args.seed
    )

    split_map = {
        "train": set(F_train),
        "val": set(F_val),
        "test": set(F_test)
    }

    # --- build splits ---
    for name, Fset in split_map.items():

        # select in-split clone pairs and add label "1"
        clone_set = []
        for p in clone_pairs:
            f1, f2 = extract_function_ids(p)
            if f1 in Fset and f2 in Fset:
                # append label column
                clone_set.append((*p, "1"))

        # select in-split non-clone pairs and add label "0"
        nonclone_set = []
        for p in nonclone_pairs:
            f1, f2 = extract_function_ids(p)
            if f1 in Fset and f2 in Fset:
                # append label column
                nonclone_set.append((*p, "0"))

        # alternate clone / non-clone
        result = []
        while clone_set and nonclone_set:
            result.append(clone_set.pop())
            result.append(nonclone_set.pop())

        # write output
        out_file = f"{name}.csv"
        write_csv(result, out_file)
        print(f"Wrote {out_file}: {len(result)} rows")


if __name__ == "__main__":
    main()
