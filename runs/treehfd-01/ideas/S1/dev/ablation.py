"""Dev only: ablation table (ortho_in / ortho_out / locvar ratio) of step 1 / step 2 variants on cached baseline fits."""
import contextlib
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dev_eval2 as D  # noqa: E402

D.N_BOOT = 1
CONFIGS = {
    "previous (m=30, argmin, no step 2)": dict(m_min=30, se_rule=0.0, close=False),
    "step 1 only": dict(close=False),
    "step 2 only": dict(se_rule=1e9),
    "final": dict(),
    "step 2 Wald-gated (z=2)": dict(close_z=2.0),
    "step 1 + directions": dict(directions=True),
    "step 1 argmin (no SE rule)": dict(se_rule=0.0),
    "bins n^(2/3)": dict(m_min=-1),
}


def fmt(x):
    return "  -  " if x is None else f"{x:.4f}"


if __name__ == "__main__":
    names = sys.argv[1:]
    print("| config | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    for label, kw in CONFIGS.items():
        cells = []
        for n in names:
            kw2 = dict(kw)
            if kw2.get("m_min") == -1:
                import numpy as np
                import pickle
                d = pickle.load(open(D.HERE / "cache" / f"{n}.pkl", "rb"))
                kw2["m_min"] = max(30, int(np.ceil(d["Xtr"].shape[0] ** (2 / 3))))
            with contextlib.redirect_stdout(io.StringIO()):
                r = D.run(n, **kw2)
            cells.append(f"{fmt(r['new'][0])} / {fmt(r['new'][1])} / {r['new'][2] / r['base'][2]:.2f}x")
        print(f"| {label} | " + " | ".join(cells) + " |", flush=True)
