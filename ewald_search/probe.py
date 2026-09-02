#!/usr/bin/env python3
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

path = Path("smooth_fano_8d.gz")
sha = hashlib.sha256(path.read_bytes()).hexdigest()

head = []
blocks = 0
facet_counts = Counter()
dim_counts = Counter()
current_rows = 0
in_block = False

with gzip.open(path, "rt", encoding="utf-8", errors="strict") as f:
    for line_no, line in enumerate(f, 1):
        if len(head) < 120:
            head.append(line)
        stripped = line.strip()
        if not in_block:
            if not stripped:
                continue
            blocks += 1
            current_rows = 0
            in_block = True
            continue
        if not stripped:
            facet_counts[current_rows] += 1
            in_block = False
            continue
        vals = [int(v) for v in stripped.split()]
        dim_counts[len(vals) - 1] += 1
        current_rows += 1

if in_block:
    facet_counts[current_rows] += 1

Path("probe_head.txt").write_text("".join(head), encoding="utf-8")
summary = {
    "compressed_bytes": path.stat().st_size,
    "sha256": sha,
    "blocks": blocks,
    "facet_count_distribution": dict(sorted(facet_counts.items())),
    "facet_row_dimension_distribution": dict(sorted(dim_counts.items())),
}
Path("probe.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
