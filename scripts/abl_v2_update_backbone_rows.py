#!/usr/bin/env python3
"""Drop fresh backbone-throughput rows into sections/results.tex.

Replaces whatever sits between the BACKBONE_ROWS_BEGIN / _END markers, so the
table body can be regenerated from the sweep logs without hand-editing tex.

    abl_v2_update_backbone_rows.py <results.tex> <fp32_log> [fp16_log]
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

BEGIN = "% BACKBONE_ROWS_BEGIN"
END = "% BACKBONE_ROWS_END"


def main() -> int:
    tex, logs = Path(sys.argv[1]), sys.argv[2:]
    gen = Path(__file__).with_name("abl_v2_backbone_table.py")
    rows = subprocess.run([sys.executable, str(gen), *logs],
                          capture_output=True, text=True, check=True).stdout
    s = tex.read_text()
    i, j = s.index(BEGIN), s.index(END)
    head = s[:i + len(BEGIN)]
    s = head + " (generated: scripts/abl_v2_backbone_table.py)\n" + rows + s[j:]
    tex.write_text(s)
    print(rows, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
