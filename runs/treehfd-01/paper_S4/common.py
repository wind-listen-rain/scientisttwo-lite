import json, math, os
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
def load(n): return json.load(open(os.path.join(R, n)))
BASE = load("baseline_full.json"); FULL = load("method_full.json")
ABL = {"no_shrinkage_direct_solver_only": "No shrinkage (kappa=0.01 only)",
       "no_lattice_prior_ridge_zero_only": "No lattice prior / harmonic rule",
       "no_ensemble_level_kappa_step": "No ensemble-level kappa step",
       "soft_orthogonality_rows": "Soft orthogonality rows"}
ABLD = {k: load(f"ablation_{k}.json") for k in ABL}
DS = list(FULL["real"].keys())
METR = ["resid_in", "resid_out", "ortho_in", "ortho_out", "locvar_in"]
AMET = ["mse_eta1","mse_eta2","mse_eta3","mse_eta4","mse_eta5","mse_eta6","mse_eta12","mse_eta34","mse_others","resid_out"]
def gm(xs):
    xs = [x for x in xs if x is not None and x > 0]
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else None
def ratio(a, b):
    return None if a is None or b is None or b == 0 else a / b
def fmt(x, d=4):
    if x is None: return "n/a"
    return f"{x:.{d}g}"
def fr(x): return "n/a" if x is None else f"{x:.2f}"
