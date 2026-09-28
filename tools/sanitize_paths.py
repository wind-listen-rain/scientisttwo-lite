"""发布前把本机绝对路径换成可移植的写法（可重复运行）。

- 文本记录（.md/.json/.jsonl/.log/.out/.txt）：本机仓库路径 → "<ROOT>"
- 智能体写的诊断脚本（.py）：字符串里的本机仓库路径 → 由脚本自身位置推算的仓库根目录
- bench/BENCHMARK.md 属于评测协议清单：改动后同步更新各次运行 state.json 里的哈希，并记入 protocol_changes
编排器读取 state.json 时会把 "<ROOT>" 换回当前仓库路径，所以清理后仍可断点续跑。
用法: python tools/sanitize_paths.py [--old 旧的仓库绝对路径]（默认取本仓库当前位置）
"""
import argparse
import hashlib
import json
import os
import stat
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT = {".md", ".json", ".jsonl", ".log", ".out", ".txt"}
SKIP_DIRS = {".conda", ".git", ".wsgit", "__pycache__", "dry"}
SKIP_TOP = {"tasks"}  # 第三方代码与论文原文，不发布
OWN_CODE = {"orchestrator.py", "tools", "bench", "verify", "prompts"}  # 我们自己的代码已改为运行时定位


def walk():
    for dirpath, dirnames, filenames in os.walk(ROOT):
        rel = Path(dirpath).relative_to(ROOT)
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not (rel == Path(".") and d in SKIP_TOP)]
        for f in filenames:
            yield Path(dirpath) / f


def writable(path, fn):
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IWUSR)
    try:
        fn()
    finally:
        path.chmod(mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default=None)
    old = ap.parse_args().old or str(ROOT)
    changed = []
    bench_doc = ROOT / "bench" / "BENCHMARK.md"
    old_hash = hashlib.sha256(bench_doc.read_bytes()).hexdigest()
    for f in walk():
        if f.suffix in TEXT:
            t = f.read_text(errors="replace")
            if old in t:
                writable(f, lambda f=f, t=t: f.write_text(t.replace(old, "<ROOT>")))
                changed.append(f)
        elif f.suffix == ".py" and f.relative_to(ROOT).parts[0] not in OWN_CODE:
            t = f.read_text()
            if old in t:
                k = len(f.relative_to(ROOT).parts) - 1
                root_expr = f'str(__import__("pathlib").Path(__file__).resolve().parents[{k}])'
                t = t.replace(f'"{old}"', root_expr).replace(f'"{old}/', f'{root_expr} + "/')
                f.write_text(t)
                changed.append(f)
    new_hash = hashlib.sha256(bench_doc.read_bytes()).hexdigest()
    if new_hash != old_hash:
        for sp in ROOT.glob("runs/*/state.json"):
            s = json.loads(sp.read_text())
            key = "bench/BENCHMARK.md"
            if s.get("manifest", {}).get(key) == old_hash:
                s["manifest"][key] = new_hash
                s.setdefault("protocol_changes", []).append({
                    "t": time.strftime("%Y-%m-%d %H:%M:%S"), "file": key, "old_sha256": old_hash, "new_sha256": new_hash,
                    "reason": "发布前把自测命令里的本机绝对路径换成 <ROOT>；不影响评测逻辑与数据"})
                sp.write_text(json.dumps(s, indent=1, ensure_ascii=False))
    for sp in ROOT.glob("runs/*/state.json"):
        s = json.loads(sp.read_text())
        if s.get("root") != "<ROOT>":
            s["root"] = "<ROOT>"
            sp.write_text(json.dumps(s, indent=1, ensure_ascii=False))
    print(f"改写 {len(changed)} 个文件")
    left = [str(f.relative_to(ROOT)) for f in walk() if f.suffix in TEXT | {".py"} and old in f.read_text(errors="replace")
            and f.name != "sanitize_paths.py"]
    print("仍含本机路径的文件：", left or "无")


if __name__ == "__main__":
    main()
