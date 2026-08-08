"""Split gbar_packed_jobs.csv into short (<=24h) and long (>24h) subsets."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "scripts/gbar_packed_jobs.csv"


def main():
    short = []
    long_ = []
    with open(SRC) as f:
        r = csv.DictReader(f)
        headers = r.fieldnames
        for row in r:
            wt = int(row["walltime_hr"])
            if wt <= 24:
                short.append(row)
            else:
                long_.append(row)

    for name, subset in [("short_v100", short), ("long_a100", long_)]:
        out = ROOT / f"scripts/gbar_packed_jobs_{name}.csv"
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=headers, lineterminator="\n", delimiter="|", quoting=csv.QUOTE_NONE)
            w.writeheader()
            for i, row in enumerate(subset, start=1):
                row = dict(row)
                row["idx"] = i
                w.writerow(row)
        print(f"Wrote {out}  ({len(subset)} rows)")


if __name__ == "__main__":
    main()
