# reorder_label.py
input_file = "datasets/training_data_500k_50k_50k/dev.csv"
output_file = "datasets/training_data_500k_50k_50k/dev2.csv"

with open(input_file, "r") as fin, open(output_file, "w") as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        label = line[0]             # first character (0 or 1)
        rest = line[1:]             # rest of the line
        fout.write(f"{rest},{label}\n")
