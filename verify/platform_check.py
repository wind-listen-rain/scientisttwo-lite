"""跨平台比对：同一份评测协议（bench/ 逐字节相同），Mac 与 Windows 上的官方子集结果差多少。

用法: python verify/platform_check.py  → 打印比对表并写出 verify/platform/compare.json
输入：baseline/subset.json 与 runs/treehfd-01/ideas/S4/subset_1.json（Mac，2026-09-27）；
      verify/platform/*_win.json（Windows，2026-09-28，同一份代码与数据）。
结论见 docs/JOURNAL.md 2026-09-28 下午：真实数据上的 XGBoost 模型两边一致，确定性的 S4 在真实数据上逐位一致；
合成数据（multivariate_normal 的 SVD 基底由 LAPACK 决定）和 locvar_in（一维近邻有并列值时的取舍）跟着平台变。
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAIRS = {"baseline": ("baseline/subset.json", "verify/platform/baseline_subset_win.json"),
         "S4": ("runs/treehfd-01/ideas/S4/subset_1.json", "verify/platform/S4_subset_win.json")}


def flat(r):
    out = {f"analytical.{k}": v for k, v in r.get("analytical", {}).get("mean", {}).items()}
    for d, m in r.get("real", {}).items():
        out.update({f"{d}.{k}": v for k, v in m.items() if isinstance(v, float)})
    return {k: v for k, v in out.items() if not k.endswith("fit_s")}


def main():
    res = {}
    for name, (mac, win) in PAIRS.items():
        a = flat(json.loads((ROOT / mac).read_text(encoding="utf-8")))
        b = flat(json.loads((ROOT / win).read_text(encoding="utf-8")))
        res[name] = {k: {"mac": a[k], "win": b.get(k), "rel_pct": None if not a[k] or b.get(k) is None
                         else (b[k] - a[k]) / abs(a[k]) * 100} for k in a}
        print(f"== {name}（Windows 相对 Mac）")
        for k, v in res[name].items():
            print(f"  {k:28s} {v['mac']:.6g} → {v['win']:.6g}  {v['rel_pct']:+.4f}%")
    (ROOT / "verify" / "platform" / "compare.json").write_text(json.dumps(res, indent=1), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
