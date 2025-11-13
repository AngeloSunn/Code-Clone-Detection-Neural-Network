import csv
import sys
from typing import List, Tuple


def read_rows(paths: List[str]) -> List[List[str]]:
    rows: List[List[str]] = []
    for path in paths:
        with open(path, newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                if len(row) != 9:
                    raise ValueError(
                        f"Expected 9 columns per row, got {len(row)} in file {path}: {row}"
                    )
                rows.append(row)
    return rows


def extract_functions(rows: List[List[str]]) -> List[Tuple[str, int, int]]:
    funcs = set()
    for r in rows:
        funcs.add((r[1], int(r[2]), int(r[3])))
        funcs.add((r[5], int(r[6]), int(r[7])))

    return sorted(funcs, key=lambda x: (x[0], x[1], x[2]))


def split_functions(funcs: List[Tuple[str, int, int]]):
    mid = len(funcs) // 2
    return set(funcs[:mid]), set(funcs[mid:])


def assign_rows_to_halves(rows, first_half, second_half):
    result = {
        "first": {"pos": [], "neg": []},
        "second": {"pos": [], "neg": []},
    }

    for r in rows:
        f1 = (r[1], int(r[2]), int(r[3]))
        f2 = (r[5], int(r[6]), int(r[7]))
        label = r[8]

        in_first = f1 in first_half and f2 in first_half
        in_second = f1 in second_half and f2 in second_half

        if not (in_first or in_second):
            continue

        bucket = "first" if in_first else "second"

        if label == "1":
            result[bucket]["pos"].append(r)
        elif label == "0":
            result[bucket]["neg"].append(r)
        else:
            raise ValueError(f"Unexpected label {label!r}")

    return result


def interleave_strict(pos_list: List[List[str]], neg_list: List[List[str]]) -> List[List[str]]:
    """
    Strictly alternate 1-label and 0-label rows.
    Stop immediately when one of the lists is exhausted.
    Ensures perfect 50/50 balance.
    """
    output: List[List[str]] = []
    length = min(len(pos_list), len(neg_list))

    for i in range(length):
        output.append(pos_list[i])
        output.append(neg_list[i])

    return output


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    if len(argv) != 4:
        print(
            "Usage: python script.py input1.csv input2.csv "
            "output_first_half.csv output_second_half.csv"
        )
        sys.exit(1)

    in1, in2, out1, out2 = argv

    rows = read_rows([in1, in2])
    funcs = extract_functions(rows)

    if not funcs:
        print("No functions found.")
        sys.exit(1)

    first_half, second_half = split_functions(funcs)
    assigned = assign_rows_to_halves(rows, first_half, second_half)

    first_rows = interleave_strict(assigned["first"]["pos"], assigned["first"]["neg"])
    second_rows = interleave_strict(assigned["second"]["pos"], assigned["second"]["neg"])

    with open(out1, "w", newline="") as f:
        csv.writer(f).writerows(first_rows)

    with open(out2, "w", newline="") as f:
        csv.writer(f).writerows(second_rows)


if __name__ == "__main__":
    main()
