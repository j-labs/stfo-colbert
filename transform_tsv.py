#!/usr/bin/env python3
import sys

if len(sys.argv) != 3:
    print("Usage: python transform_tsv.py input.tsv output.txt")
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]

with open(input_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

seen = set()
results = []
duplicates = 0

for line in lines:
    line = line.rstrip('\n')
    if line:
        cols = line.split('\t')
        if len(cols) >= 2:
            combined = f"{cols[0]} | {cols[1]}"
            if combined not in seen:
                seen.add(combined)
                results.append(combined)
            else:
                duplicates += 1

with open(output_file, 'w', encoding='utf-8') as f:
    f.write("\n\n--------\n\n".join(results))

print(f"Transformed {len(results)} unique rows from {input_file} to {output_file}")
if duplicates > 0:
    print(f"Removed {duplicates} duplicate(s)")
