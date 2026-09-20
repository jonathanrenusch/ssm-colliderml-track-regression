#!/usr/bin/env python3
"""LaTeX body for the backbone-throughput table from the sweep logs.

    abl_v2_backbone_table.py <fp32_log> [fp16_log]

Reads scripts/abl_v2_backbone_throughput.sh output (### header lines followed
by the bench's throughput/VRAM lines) and prints the tabular rows: one row per
encoder, columns = reference PyTorch implementation / this campaign's fused
path, plus the fp16 column when the second log is given.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

HDR = re.compile(r"^### (?P<tag>\S+) mode=(?P<mode>\w+) bs=(?P<bs>\d+) dtype=(?P<dt>\S+)")
THR = re.compile(r"throughput\s*:\s*([\d,]+)")
LABEL = {
    "mamba2_bidir": r"Mamba-2, bidirectional",
    "transformer": r"Transformer",
    "mingru": r"minGRU",
    "diagssm_nonsel": r"diagonal SSM, non-selective",
}
ORDER = ["mingru", "mamba2_bidir", "transformer", "diagssm_nonsel"]


def parse(path: Path):
    out, cur = {}, None
    for line in path.read_text().splitlines():
        m = HDR.match(line)
        if m:
            cur = (m.group("tag"), m.group("mode"))
            continue
        t = THR.search(line)
        if t and cur:
            out[cur] = int(t.group(1).replace(",", ""))
            cur = None
    return out


def fmt(v):
    # A missing cell means the configuration does not run at all, not that the
    # measurement is outstanding: the only one is Mamba-2 under fp16, whose
    # fused gating kernel is typed fp32/fp64 ("Expected dtype ['fp32','fp64']
    # but got fp16").  The caption says so.
    return f"{v/1e6:.2f}" if v else "---"


def main() -> int:
    a32 = parse(Path(sys.argv[1]))
    a16 = parse(Path(sys.argv[2])) if len(sys.argv) > 2 else {}
    for tag in ORDER:
        cells = [fmt(a32.get((tag, "ref"))), fmt(a32.get((tag, "opt")))]
        if a16:
            cells.append(fmt(a16.get((tag, "opt"))))
        print(f"    {LABEL[tag]:<30s} & " + " & ".join(cells) + r" \\")
    return 0


if __name__ == "__main__":
    sys.exit(main())
