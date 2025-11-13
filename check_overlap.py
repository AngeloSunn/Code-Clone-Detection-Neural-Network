import csv
import sys
from typing import Set, Tuple, List


Function = Tuple[str, int, int]


def read_functions_from_csv(path: str) -> Set[Function]:
    """
    Read all unique functions from a CSV of layout:
    selected,FILE,START,END,selected,FILE2,START2,END2,LABEL

    Returns a set of (filename, start, end) tuples.
    """
    funcs: Set[Function] = set()

    with open(path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if len(row) != 9:
                raise ValueError(
                    f"Expected 9 columns per row in {path}, got {len(row)}: {row}"
                )

            # First function
            funcs.add((row[1], int(row[2]), int(row[3])))
            # Second function
            funcs.add((row[5], int(row[6]), int(row[7])))

    return funcs


def main(argv: List[str] = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    if len(argv) != 2:
        print("Usage: python check_function_overlap.py file1.csv file2.csv")
        sys.exit(1)

    file1, file2 = argv

    funcs1 = read_functions_from_csv(file1)
    funcs2 = read_functions_from_csv(file2)

    overlap = funcs1 & funcs2

    print(f"File 1: {file1}")
    print(f"  Unique functions: {len(funcs1)}")
    print(f"File 2: {file2}")
    print(f"  Unique functions: {len(funcs2)}")
    print()
    print(f"Overlap (same (file, start, end) in both): {len(overlap)}")

    if overlap:
        print("\nOverlapping functions:")
        for fname, start, end in sorted(overlap, key=lambda x: (x[0], x[1], x[2])):
            print(f"{fname}:{start}-{end}")

    # Optional: non-zero exit code if there is overlap
    # sys.exit(1 if overlap else 0)


if __name__ == "__main__":
    main()
