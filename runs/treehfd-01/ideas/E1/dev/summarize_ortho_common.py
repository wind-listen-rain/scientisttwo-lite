"""Geo-mean E1 / S4 of the harness ortho and of the common-set ortho (dev/logs/diag_ortho_common_*.log).

usage: summarize_ortho_common.py <dataset> [...]"""
import re
import sys
from pathlib import Path

import numpy as np

pat = re.compile(r"split (\d+) (in|out)\s*: harness S4=(\S+) E1=(\S+) \| common set \(n=(\d+)\) S4=(\S+) "
                 r"E1=(\S+) median pair ratio=(\S+) \| n_big S4=(\d+) E1=(\d+) \| inter share S4=(\S+) E1=(\S+)")
for ds in sys.argv[1:]:
    rows = [pat.search(l).groups() for l in
            (Path(__file__).parent / "logs" / f"diag_ortho_common_{ds}.log").read_text().splitlines()
            if pat.search(l)]
    for lab in ("in", "out"):
        r = [x for x in rows if x[1] == lab]
        def gm(a, b):
            v = [float(x[b]) / float(x[a]) for x in r
                 if x[a] != "nan" and x[b] != "nan" and float(x[a]) > 0]
            return np.exp(np.mean(np.log(v))), len(v)
        h, nh = gm(2, 3)
        c, nc = gm(5, 6)
        s, _ = gm(10, 11)
        med = np.median([float(x[7]) for x in r if x[7] != "nan"])
        nb4 = sum(int(x[8]) for x in r); nb1 = sum(int(x[9]) for x in r)
        print(f"{ds:9s} ortho_{lab:3s} E1/S4: harness {h:.3f} (n={nh}) | common pair set {c:.3f} (n={nc}) "
              f"| median per-pair ratio {med:.3f} | pairs >=1%: S4 {nb4} E1 {nb1} | interaction "
              f"variance share {s:.3f}")
