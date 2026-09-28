"""GitHub 存档与同步、订阅用量查询、路径可移植（orchestrator.py 与 tools/autopilot.py 共用）。

为什么需要：流水线走 Claude 订阅额度，额度用尽时智能体调用会被打断，一跑就是好几天，中途也可能换机器接着跑。
约定（2026-09-28 起）：
- 每完成一个步骤、每 30 分钟、用量到 90% 时、撞到用量上限准备等待前，都把整个仓库提交并推送到 GitHub（存档）；
- 等到额度重置后，先看 GitHub 上有没有别的机器推的新进度；有就按新进度来（编排器退出，由 autopilot 同步后重启）；
- 写进仓库的文本里，本机仓库路径一律换成 "<ROOT>"（编排器读 state.json 时会换回本机路径），换机器也能直接续跑。
"""
import json
import os
import re
import socket
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOST = socket.gethostname()
REMOTE_EXIT = 75  # 编排器发现 GitHub 上有新进度时的退出码
LOCK = threading.RLock()
TEXT = {".md", ".json", ".jsonl", ".log", ".out", ".txt", ".csv", ".tsv"}
TRAILER = "\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"


def git(*args, cwd=ROOT, timeout=600):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout)


# ---------- 路径可移植 ----------
def _root_forms():
    s, p = str(ROOT), ROOT.as_posix()
    forms = {s, p, json.dumps(s)[1:-1]}
    if os.name == "nt" and len(p) > 2 and p[1] == ":":
        forms |= {f[0].swapcase() + f[1:] for f in list(forms)}
        forms |= {"/" + p[0].lower() + p[2:], "/" + p[0].upper() + p[2:]}  # Git Bash 写法 /d/...
    return sorted(forms, key=len, reverse=True)


FORMS = _root_forms()


def portable(text):
    """文本里的本机仓库路径 → "<ROOT>"。"""
    for f in FORMS:
        text = text.replace(f, "<ROOT>")
    return text


def _portable_py(src, depth):
    """Python 源码里以本机仓库路径开头的字符串字面量 → 由文件自身位置推算的仓库根目录（与 sanitize_paths.py 同一规则）。"""
    expr = f'str(__import__("pathlib").Path(__file__).resolve().parents[{depth}])'
    for f in FORMS:
        pat = re.compile(r"(?<![\w\"'])(?P<pre>[rRbBfFuU]{0,2})(?P<q>[\"'])" + re.escape(f) + r"(?P<end>(?P=q))?")
        src = pat.sub(lambda m: expr if m["end"] else f"{expr} + {m['pre']}{m['q']}", src)
    return src


def sanitize_tree(top):
    """把一个目录（工作区、论文目录）里写进了本机路径的文件改成可移植写法；改完的 .py 编译不过就保留原样。"""
    top = Path(top)
    for dirpath, dirnames, filenames in os.walk(top):
        dirnames[:] = [d for d in dirnames if d not in (".wsgit", ".git", "__pycache__")]
        for name in filenames:
            f = Path(dirpath) / name
            if f.suffix not in TEXT | {".py"}:
                continue
            try:
                raw = f.read_bytes()
                t = raw.decode("utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if not any(x in t for x in FORMS):
                continue
            if f.suffix == ".py":
                new = _portable_py(t, len(f.resolve().relative_to(ROOT).parts) - 1)
                try:
                    compile(new, str(f), "exec")
                except SyntaxError:
                    continue
            else:
                new = portable(t)
            if new != t:
                try:
                    f.write_bytes(new.encode("utf-8"))
                except OSError:
                    pass


# ---------- 工作区版本历史（<工作区>/.wsgit ↔ HISTORY.bundle） ----------
def ws_dirs():
    found = list(ROOT.glob("runs/*/*/.wsgit")) + list(ROOT.glob("runs/*/ideas/*/.wsgit"))
    return sorted(d.parent for d in found if d.relative_to(ROOT).parts[1] != "dry")  # runs/dry 是空跑产物，不发布


def _bundle_head(b):
    if not b.exists():
        return None
    for line in git("bundle", "list-heads", str(b)).stdout.splitlines():
        sha, _, ref = line.partition(" ")
        if ref == "HEAD":
            return sha
    return None


def refresh_bundles():
    """工作区历史有新提交时，重新生成它的 HISTORY.bundle（仓库里发布的是 bundle，.wsgit 不进外层仓库）。"""
    for d in ws_dirs():
        gd = f"--git-dir={d / '.wsgit'}"
        head = git(gd, "rev-parse", "HEAD").stdout.strip()
        b = d / "HISTORY.bundle"
        if head and head != _bundle_head(b):
            tmp = d / "HISTORY.bundle.tmp"
            if git(gd, "bundle", "create", str(tmp), "--all").returncode == 0:
                tmp.replace(b)


def restore_ws_histories(log=print):
    """按仓库里的 HISTORY.bundle 还原或更新各工作区的 .wsgit（新克隆、或采用了别的机器的进度之后）。"""
    for b in sorted(ROOT.glob("runs/*/ideas/*/HISTORY.bundle")) + sorted(ROOT.glob("runs/*/paper_*/HISTORY.bundle")):
        d, gdir = b.parent, b.parent / ".wsgit"
        head = _bundle_head(b)
        gd = f"--git-dir={gdir}"
        if not gdir.exists():
            git("-c", "core.autocrlf=false", "clone", "-q", "--bare", str(b), str(gdir))
            git(gd, "config", "core.bare", "false")
        elif head and head != git(gd, "rev-parse", "HEAD").stdout.strip():
            git(gd, "fetch", "-q", "--update-head-ok", str(b), "+refs/heads/*:refs/heads/*")
        else:
            continue
        git(gd, "config", "core.autocrlf", "false")
        (gdir / "info").mkdir(exist_ok=True)
        (gdir / "info" / "exclude").write_text(".wsgit/\nHISTORY.bundle\n__pycache__/\n", encoding="utf-8")
        git(gd, f"--work-tree={d}", "reset", "-q", "HEAD")  # 只更新索引；工作区文件以外层仓库为准
        log(f"还原工作区历史：{d.relative_to(ROOT).as_posix()} → {head and head[:7]}")


# ---------- 订阅用量 ----------
def usage():
    """订阅用量（与 Claude Code 的 /usage 同一个接口）。拿不到（比如 macOS 的令牌在钥匙串里）就返回 None。
    令牌只发给 api.anthropic.com，不打印、不落盘。"""
    try:
        cred = json.loads((Path.home() / ".claude" / ".credentials.json").read_text(encoding="utf-8"))
        tok = cred["claudeAiOauth"]["accessToken"]
        req = urllib.request.Request("https://api.anthropic.com/api/oauth/usage", headers={
            "Authorization": f"Bearer {tok}", "anthropic-beta": "oauth-2025-04-20", "User-Agent": "claude-code/2.1.283"})
        with urllib.request.urlopen(req, timeout=20) as r:
            d = json.loads(r.read().decode("utf-8"))
    except Exception:  # noqa: BLE001 — 查不到用量不影响主流程
        return None
    out = {k: {"pct": d[k]["utilization"], "resets_at": d[k].get("resets_at")}
           for k in ("five_hour", "seven_day", "seven_day_opus", "seven_day_sonnet")
           if isinstance(d.get(k), dict) and d[k].get("utilization") is not None}
    return out or None


def _epoch(iso):
    from datetime import datetime
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()


MONTHS = {m: i for i, m in enumerate(("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"), 1)}


def seconds_until_reset(text):
    """从 "resets 10:40pm"、"resets Oct 3, 9am" 这类提示里算出还要等多少秒；解析不了返回 None。"""
    m = re.search(r"resets\s+(?:(?P<mon>[A-Za-z]{3})[a-z]*\.?\s+(?P<day>\d{1,2}),?\s+(?:at\s+)?)?"
                  r"(?P<h>\d{1,2})(?::(?P<mi>\d{2}))?\s*(?P<ap>am|pm)", text or "", re.I)
    if not m:
        return None
    h, mi = int(m["h"]) % 12 + (12 if m["ap"].lower() == "pm" else 0), int(m["mi"] or 0)
    now = time.localtime()
    if m["mon"] and m["mon"][:3].lower() in MONTHS:
        mon, day = MONTHS[m["mon"][:3].lower()], int(m["day"])
        target = time.mktime((now.tm_year, mon, day, h, mi, 0, 0, 0, -1))
        if target < time.time() - 86400:
            target = time.mktime((now.tm_year + 1, mon, day, h, mi, 0, 0, 0, -1))
    else:
        target = time.mktime((now.tm_year, now.tm_mon, now.tm_mday, h, mi, 0, 0, 0, -1))
        if target <= time.time():
            target += 86400
    return max(0.0, target - time.time())


def reset_wait(text):
    """撞到上限后要等多少秒：优先用用量接口里已用满的窗口的重置时间，其次解析提示文字，都不行就 30 分钟。"""
    u = usage() or {}
    full = [_epoch(v["resets_at"]) for v in u.values() if v.get("pct", 0) >= 99 and v.get("resets_at")]
    if full:
        return max(0.0, max(full) - time.time())
    s = seconds_until_reset(text)
    return 1800.0 if s is None else s


# ---------- 心跳 ----------
def read_runner(run_dir):
    try:
        return json.loads((Path(run_dir) / "RUNNER.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_runner(run_dir, status, **extra):
    """runs/<运行>/RUNNER.json：哪台机器、什么状态、最后一次活着的时间（续跑时据此扣掉停机时长，也用来避免两台机器同时跑）。"""
    info = {"host": HOST, "pid": os.getpid(), "status": status, "last_alive": time.time(),
            "last_alive_local": time.strftime("%Y-%m-%d %H:%M:%S"), **extra}
    p = Path(run_dir) / "RUNNER.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(info, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    tmp.replace(p)


def other_runner_active(run_dir, fresh_s=2700):
    """GitHub 同步下来的 RUNNER.json 显示另一台机器正在跑、且 45 分钟内有过心跳 → 返回它的信息。"""
    r = read_runner(run_dir)
    if r and r.get("host") != HOST and r.get("status") == "running" and time.time() - r.get("last_alive", 0) < fresh_s:
        return r
    return None


# ---------- 存档与同步 ----------
def checkpoint(msg, push=True):
    """提交整个仓库并推送到 GitHub。返回 "pushed" / "nothing" / "remote_ahead"（GitHub 上有本机没有的新提交）/ "push_failed"。"""
    with LOCK:
        refresh_bundles()
        git("add", "-A")
        if git("diff", "--cached", "--quiet").returncode != 0:
            git("commit", "-q", "-m", f"存档（{HOST}）：{msg}{TRAILER}")
        if not push:
            return "committed"
        if git("rev-list", "--count", "origin/main..HEAD").stdout.strip() == "0":
            return "nothing"
        for attempt in range(3):
            r = git("push", "-q", "origin", "HEAD:main", timeout=300)
            if r.returncode == 0:
                git("fetch", "-q", "origin", timeout=300)
                return "pushed"
            if re.search(r"rejected|non-fast-forward|fetch first", r.stderr):
                return "remote_ahead"
            time.sleep(20 * (attempt + 1))
        return "push_failed"


def remote_news():
    """GitHub 上有没有本机还没有的新提交（别的机器或别人推的进度）。网络不通时返回 None。"""
    with LOCK:
        if git("fetch", "-q", "origin", timeout=300).returncode != 0:
            return None
        n = git("rev-list", "--count", "HEAD..origin/main").stdout.strip()
        return int(n or 0) > 0


def adopt_remote(log=print):
    """按 GitHub 上的新进度来：能快进就快进；两边都有新提交时，本机提交存到 backup/ 分支（也推上去）后对齐 GitHub。"""
    with LOCK:
        git("add", "-A")
        if git("diff", "--cached", "--quiet").returncode != 0:
            git("commit", "-q", "-m", f"存档（{HOST}）：采用 GitHub 新进度前的本机快照{TRAILER}")
        if git("merge-base", "--is-ancestor", "HEAD", "origin/main").returncode == 0:
            r = git("merge", "-q", "--ff-only", "origin/main")
            log(f"GitHub 有新进度，已快进到 {git('rev-parse', '--short', 'HEAD').stdout.strip()}"
                + ("" if r.returncode == 0 else f"（失败：{r.stderr.strip()[:300]}）"))
        else:
            br = f"backup/{HOST}-{time.strftime('%Y%m%d-%H%M%S')}"
            git("branch", br)
            git("push", "-q", "origin", br, timeout=300)
            r = git("reset", "-q", "--hard", "origin/main")
            log(f"GitHub 与本机都有新提交：本机进度已存到分支 {br}，按 GitHub 的进度继续"
                + ("" if r.returncode == 0 else f"（对齐失败：{r.stderr.strip()[:300]}）"))
        restore_ws_histories(log)
