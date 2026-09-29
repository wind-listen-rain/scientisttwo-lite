"""One real-data job of the rebuttal experiments (R1, R2, R3, R5-i); label-free decompositions.

usage: run_real.py <job> <dataset>
  job = core        baseline + submitted GT-LOCO (re-fit here), per-point test arrays (R1), R2 (d) purified
                    baseline and (e) lattice+harmonic with one global kappa, R3 shift-range sweep,
                    R5(i) fixed-shift sweep
  job = ridge       R2 (a) ridge/zero and (b) ridge/harmonic, one global kappa chosen by the leave-out risk
  job = cfg:<name>  R3 refit with one changed constant (lattice ridge, main-bin ridge, kappa grid)
Results: supp/raw/<job>__<dataset>.json (+ .npz arrays for `core`).
"""
import sys
import time

import numpy as np

from common import (Out, SUPP, SweepHFD, baseline_fit, dump, gm, out_from_baseline, purify, real,
                    real_metrics, set_config)

CFGS = {
    "lat0.03": dict(lattice_ridge=0.03), "lat0.3": dict(lattice_ridge=0.3),
    "main0.3": dict(eps_main=0.3), "main3": dict(eps_main=3.0),
    "kappa6": dict(kappas=np.geomspace(1e-2, 1e4, 6)), "kappa24": dict(kappas=np.geomspace(1e-2, 1e4, 24)),
}
SHIFT_SETS = {"0..0": range(0, 1), "-3..+1": range(-3, 2), "-5..+3": range(-5, 4), "-8..+5": range(-8, 6)}


def evalc(sw, cand, model, Xtr, Xte):
    sw.apply(cand)
    o = sw.out(Xtr, Xte)
    m = real_metrics(model, Xtr, Xte, o)
    m.update(cand=[cand[0], int(cand[1]), float(cand[2])], risk_over_var=cand[3] / sw.var_t,
             fid_over_var=cand[4] / sw.var_t)
    return m, o


def main(job, ds):
    model, Xtr, Xte = real(ds)
    res = {"dataset": ds, "job": job, "n_train": len(Xtr), "n_test": len(Xte)}
    if job == "core":
        set_config()
        hb, tb = baseline_fit(model, Xtr)
        ob = out_from_baseline(hb, Xtr, Xte)
        res["baseline"] = real_metrics(model, Xtr, Xte, ob)
        res["baseline"]["fit_s"] = tb
        # the baseline's unseen-cell tie-break is random: 5 further draws of its test predictions
        res["baseline_redraws"] = []
        for _ in range(5):
            m_, i_ = hb.predict(Xte, verbose=False)
            o_ = Out(ob.eta0, ob.il, Xtr, Xte, lambda X, m_=m_, i_=i_, ob=ob: ob.tr if X is Xtr else (m_, i_))
            res["baseline_redraws"].append(real_metrics(model, Xtr, Xte, o_))
        pur = purify(ob, Xtr, Xte)
        res["purified_baseline"] = real_metrics(model, Xtr, Xte, pur)
        t0 = time.perf_counter()
        sw = SweepHFD(model)
        sw.fit(Xtr)
        res["sweep_fit_s"] = time.perf_counter() - t0
        default = sw.choose()
        res["default"], og = evalc(sw, default, model, Xtr, Xte)
        v0 = default[1]
        # R2 (e): lattice prior + unseen-cell rule chosen as in the method, but ONE global kappa
        # (no per-tree kappa, no shift): candidates = common kappa only.
        res["lattice_common"], _ = evalc(sw, sw.choose(shift_cands=False), model, Xtr, Xte)
        # R3: shift range (selection among shifts in the set and the common kappas, as in the method)
        res["shift_range"] = {}
        for name, rng in SHIFT_SETS.items():
            res["shift_range"][name], _ = evalc(sw, sw.choose(shifts=rng), model, Xtr, Xte)
        # R5(i): fixed shifts with the selected variant, no selection
        res["shift_sweep"] = {}
        for s in range(-5, 4):
            cands = {c[2]: c for c in sw.candidates(shifts=[s], common=False, variants=[v0])}
            res["shift_sweep"][str(s)], _ = evalc(sw, cands[s], model, Xtr, Xte)
        res["selected_variant"] = int(v0)
        res["kappas"] = list(map(float, __import__("gtloco").KAPPAS))
        # arrays for the test-row bootstrap (R1): the same fitted decompositions for both methods
        sw.apply(default)
        og = sw.out(Xtr, Xte)
        np.savez_compressed(SUPP / f"raw/core__{ds}.npz", p_te=model.predict(Xte),
                            base_main=ob.te[0], base_inter=ob.te[1], base_il=ob.il, base_eta0=ob.eta0,
                            gt_main=og.te[0], gt_inter=og.te[1], gt_il=og.il, gt_eta0=og.eta0)
    elif job == "ridge":
        set_config(priors=("ridge",), rules=("zero", "harmonic"))
        sw = SweepHFD(model)
        sw.fit(Xtr)
        res["ridge_zero"], _ = evalc(sw, sw.choose(variants=[0], shift_cands=False), model, Xtr, Xte)
        res["ridge_harmonic"], _ = evalc(sw, sw.choose(variants=[1], shift_cands=False), model, Xtr, Xte)
        res["ridge_full_selection"], _ = evalc(sw, sw.choose(), model, Xtr, Xte)
    elif job.startswith("cfg:"):
        name = job[4:]
        set_config(**CFGS[name])
        sw = SweepHFD(model)
        t0 = time.perf_counter()
        sw.fit(Xtr)
        res["sweep_fit_s"] = time.perf_counter() - t0
        res["default_sel"], _ = evalc(sw, sw.choose(), model, Xtr, Xte)
        G = len(__import__("gtloco").KAPPAS)
        if G != 12:   # shift range rescaled so that it spans the same range of kappa values
            f = G / 12
            rng = range(int(round(-5 * f)), int(round(3 * f)) + 1)
            res["scaled_shift_sel"], _ = evalc(sw, sw.choose(shifts=rng), model, Xtr, Xte)
    dump(res, SUPP / f"raw/{job.replace(':', '_')}__{ds}.json")
    print("done", job, ds, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
