#!/usr/bin/env python3
"""Drop generated rows between BEGIN/END markers in a tex file.

    abl_v2_update_arch_rows.py <tex> <begin_marker> <rows_file>

Used for the encoder-ablation tables, whose bodies come from
scripts/abl_v2_arch_table.py and should never be hand-edited.
"""
from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    tex, begin, rows_file = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    end = begin.replace("_BEGIN", "_END")
    rows = "".join(l for l in rows_file.read_text().splitlines(keepends=True)
                   if l.strip() and not l.startswith("%"))
    s = tex.read_text()
    i, j = s.index(begin), s.index(end)
    tex.write_text(s[:i + len(begin)] + "\n" + rows + s[j:])
    print(f"[rows] {len(rows.splitlines())} lines -> {tex}:{begin}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
