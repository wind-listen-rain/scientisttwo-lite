from common import *
import os
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
os.makedirs("figures", exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
C = ["#4c78a8", "#e45756", "#54a24b", "#f2b134", "#7d5ba6"]
x = np.arange(len(DS))
fig, ax = plt.subplots(figsize=(8, 3.4)); w = 0.16
for i, m in enumerate(METR):
    r = [ratio(FULL["real"][d][m], BASE["real"][d][m]) for d in DS]
    ax.bar(x + (i - 2) * w, [v if v else np.nan for v in r], w, label=m, color=C[i])
ax.axhline(1, color="k", lw=0.8); ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(DS, rotation=30)
ax.set_ylabel("GT-LOCO / TreeHFD (lower is better)")
ax.legend(ncol=5, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.15), frameon=False)
plt.tight_layout(); plt.savefig("figures/fig1_real_ratio.png", dpi=160); plt.close()

fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
for ax, m in zip(axs, ["resid_out", "resid_in"]):
    vs = [("GT-LOCO", FULL)] + [(ABL[k], ABLD[k]) for k in ABL]
    for i, (n, res) in enumerate(vs):
        r = [ratio(res["real"][d][m], BASE["real"][d][m]) for d in DS]
        ax.bar(x + (i - 2) * 0.16, r, 0.16, label=n, color=C[i])
    ax.axhline(1, color="k", lw=0.8); ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(DS, rotation=30)
    ax.set_title(m + " relative to TreeHFD")
axs[0].legend(fontsize=6.5, frameon=False)
plt.tight_layout(); plt.savefig("figures/fig2_ablation.png", dpi=160); plt.close()

fig, ax = plt.subplots(figsize=(6, 3))
names = ["analytical"] + DS
r = [FULL["analytical"]["mean"]["fit_s"] / BASE["analytical"]["mean"]["fit_s"]] + [FULL["real"][d]["fit_s"] / BASE["real"][d]["fit_s"] for d in DS]
ax.bar(range(len(names)), r, color=C[0]); ax.axhline(1, color="k", lw=0.8); ax.axhline(5, color=C[1], ls="--", lw=0.8)
ax.set_ylabel("fit-time ratio (dashed: 5x limit)"); ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=30)
plt.tight_layout(); plt.savefig("figures/fig3_fit_time.png", dpi=160); plt.close()
