"""Selection diagnostics of an ablation (or method.py) on the benchmark setups.

usage: diag_selection.py <method file> [datasets=analytical,airfoil,concrete,abalone]
Loads the method file as the harness runner does, fits it on the harness's model and X_train
(analytical: repetition 0; real data: split 0), and prints the label-free diagnostics from
`state.diagnostics`: selected ensemble candidate, chosen-kappa quantiles, R / Var[T] and the
in-sample residual / Var[T]. Nothing here is used by any method. Run one method file per
process, because the ablations set module-level switches of gtloco.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dev"))
import common  # noqa: E402


def main():
    path = Path(sys.argv[1]).resolve()
    opts = dict(a.split("=") for a in sys.argv[2:] if "=" in a)
    names = opts.get("datasets", "analytical,airfoil,concrete,abalone").split(",")
    spec = importlib.util.spec_from_file_location("method_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import gtloco  # the module the method file configured
    print(f"=== {path.name}: KAPPAS={gtloco.KAPPAS.tolist()} SHIFTS={gtloco.SHIFTS.tolist()} "
          f"PRIORS={gtloco.PRIORS} RULES={gtloco.RULES} HARD_ORTHO={gtloco.HARD_ORTHO} "
          f"COMMON_KAPPA={gtloco.COMMON_KAPPA} n_candidates/variant="
          f"{len(gtloco.SHIFTS) + (len(gtloco.KAPPAS) if gtloco.COMMON_KAPPA else 0)}")
    VAR = gtloco.variants()
    for name in names:
        model, Xtr, _ = common.analytical(0) if name == "analytical" else common.real(name)
        state = mod.fit(model, Xtr)
        d = state.diagnostics
        mode, v, par = d["selection"][:3]
        pr, ru = VAR[v]
        q = np.quantile(d["kappa_chosen"], [0, 0.25, 0.5, 0.75, 1])
        best_by_var = {}
        for c in d["candidates"]:
            key = f"{gtloco.PRIORS[VAR[c[1]][0]]}/{gtloco.RULES[VAR[c[1]][1]]}"
            best_by_var[key] = min(best_by_var.get(key, np.inf), c[3])
        var_t = d["selection"][3] / d["risk_over_var"]
        print(f"{name:10s} sel={mode} {gtloco.PRIORS[pr]}/{gtloco.RULES[ru]} {par:+d} "
              f"R/var={d['risk_over_var']:.5f} resid_in/var={d['resid_in_over_var']:.5f} "
              f"kappa q0,25,50,75,100={', '.join(f'{x:g}' for x in q)} "
              f"N1/n={np.mean(d['n1_frac']):.3f}")
        print(" " * 11 + "min R/var per variant: "
              + "  ".join(f"{k}={r / var_t:.5f}" for k, r in best_by_var.items()))


if __name__ == "__main__":
    main()
