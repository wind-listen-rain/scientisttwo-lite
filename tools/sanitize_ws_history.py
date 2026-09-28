"""把工作区版本历史（.wsgit）里每个提交中的本机路径按 sanitize_paths.py 的同一规则改写，然后重新生成 HISTORY.bundle。

用法: python tools/sanitize_ws_history.py <工作区目录> [--old 旧的仓库绝对路径]
由 git filter-branch 的 --tree-filter 在每个提交的检出里调用本脚本的 --apply 模式。
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT = {".md", ".json", ".jsonl", ".log", ".out", ".txt"}


def apply(tree, depth_offset, old):
    # 编译缓存和性能分析文件不该进版本库（其中嵌有源文件绝对路径），逐个提交删除
    import shutil
    for d in list(Path(tree).rglob("__pycache__")):
        if ".wsgit" not in d.parts:
            shutil.rmtree(d, ignore_errors=True)
    for f in list(Path(tree).rglob("*.prof")):
        if ".wsgit" not in f.parts:
            f.unlink()
    for dirpath, dirnames, filenames in os.walk(tree):
        dirnames[:] = [d for d in dirnames if d not in (".wsgit", "__pycache__")]
        for name in filenames:
            f = Path(dirpath) / name
            try:
                t = f.read_text()
            except (UnicodeDecodeError, OSError):
                continue
            if old not in t:
                continue
            if f.suffix == ".py":
                k = len(f.relative_to(tree).parts) - 1 + depth_offset
                expr = f'str(__import__("pathlib").Path(__file__).resolve().parents[{k}])'
                t = t.replace(f'"{old}"', expr).replace(f'"{old}/', f'{expr} + "/')
            elif f.suffix in TEXT:
                t = t.replace(old, "<ROOT>")
            else:
                continue
            f.write_text(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ws")
    ap.add_argument("--old", default=None)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--offset", type=int, default=0)
    a = ap.parse_args()
    old = a.old or str(ROOT)
    if a.apply:
        apply(Path(a.ws), a.offset, old)
        return
    ws = Path(a.ws).resolve()
    offset = len(ws.relative_to(ROOT).parts)
    gd = ["git", f"--git-dir={ws / '.wsgit'}", f"--work-tree={ws}"]
    cmd = f'{sys.executable} {Path(__file__).resolve()} . --apply --offset {offset} --old "{old}"'
    env = {**os.environ, "FILTER_BRANCH_SQUELCH_WARNING": "1"}
    # 工作区里只允许有 sanitize_paths.py 造成的改动：先还原，改写历史后再对齐到新的 HEAD（结果相同）
    subprocess.run([*gd, "checkout", "--", "."], check=True, cwd=ws)
    subprocess.run([*gd, "filter-branch", "-f", "--tree-filter", cmd, "--", "--all"], check=True, env=env,
                   capture_output=True, text=True, cwd=ws)
    subprocess.run([*gd, "reset", "-q", "--hard", "HEAD"], check=True, cwd=ws)
    for ref in subprocess.run([*gd, "for-each-ref", "--format=%(refname)", "refs/original/"],
                              capture_output=True, text=True).stdout.split():
        subprocess.run([*gd, "update-ref", "-d", ref], check=True)
    subprocess.run([*gd, "reflog", "expire", "--expire=now", "--all"], check=True)
    subprocess.run([*gd, "gc", "-q", "--prune=now"], check=True)
    (ws / "HISTORY.bundle").unlink(missing_ok=True)
    subprocess.run([*gd, "bundle", "create", str(ws / "HISTORY.bundle"), "--all"], check=True, capture_output=True)
    status = subprocess.run([*gd, "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
    print(f"{ws.name}: 历史已改写，工作区与 HEAD 差异 = {status or '无'}")


if __name__ == "__main__":
    main()
