"""ScientistTwo-lite：按 ScientistTwo（arXiv 2609.19644）第 3 节与附录 A.2 复现的简化版自主科研流水线。

用法:
  python orchestrator.py --run <run_id> [--dry-run]
  无人值守（推荐）：python tools/autopilot.py --run <run_id>（同步 GitHub → 跑本脚本 → 存档推送，出错自动重启）
每个角色是一次独立的 `claude -p` 调用（提示词在 prompts/），审查员总在干净上下文里运行。
官方评测由本脚本调用只读的 bench/harness.py 完成，智能体自测的结果不作数。
所有中间状态存于 runs/<run_id>/state.json，中断后重跑同一命令会从断点继续。

2026-09-28 起的改动（详见 docs/JOURNAL.md）：
- 能在 Windows 上跑（解释器路径、claude.exe、提示词走标准输入、UTF-8、正斜杠路径、防休眠）；
- 评测环境指纹（tools/env_fingerprint.py）：换了机器、合成数据或近邻不同时，基线和要比较的旧结果在本机重算，只在同一环境内比较；
- 存档与同步（tools/gitsync.py）：完成步骤、每 30 分钟、用量到 90%、撞到上限等待前都推送到 GitHub；
  额度重置后先查 GitHub，有别处推的新进度就退出（退出码 75），由 tools/autopilot.py 同步后按新进度重启；
- 撞到用量上限后用 --resume 接着被打断的会话做，不从头来；停机时长自动扣除，不占 12 小时预算。
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "tools"))
import gitsync  # noqa: E402  存档/同步、用量查询、路径可移植

WIN = os.name == "nt"
PY = ROOT / ".conda" / ("python.exe" if WIN else "bin/python")
ROOT_S, PY_S = ROOT.as_posix(), PY.as_posix()  # 写进提示词、状态和日志的路径一律用正斜杠（Git Bash 与 JSON 都能直接用）
PROMPTS = ROOT / "prompts"
VERIFY = ROOT / "tools" / "verify_refs.py"

# 预算：括号内是 ScientistTwo 附录 A.2 的原始设置
CFG = {
    "lim_rounds": 3,          # 找局限最多轮数（16）
    "n_seed": 4,              # 种子想法数（未公开）
    "max_rounds": 2,          # 想法实验轮数 K（4）
    "successes": 2,           # 攒够 S 个成功想法就停（4）
    "n_eng": 2,               # 每个想法最多工程改进次数（2）
    "n_abl": 1,               # 消融后改方法次数（1）
    "n_peer": 2,              # 模拟审稿-补实验轮数（2）
    "review_threshold": 8,    # 审稿分达标线（8）
    "n_meta": 1,              # 元审稿后返工次数（1）
    "max_hours": 12,          # 全局墙钟上限（只算真正在工作的时间）
    "max_calls": 110,         # 全局智能体调用上限
    "agent_timeout": 3600,    # 单次智能体调用超时秒数
}
FAST, STRONG = "sonnet", "opus"  # 对应原文 Gemini 3.6 Flash / Claude Code + Opus 4.8
READ = ["Read", "Glob", "Grep"]
CODE = READ + ["Write", "Edit", f"Bash({PY_S}:*)", "Bash(python:*)", "Bash(ls:*)", "Bash(cat:*)", "Bash(head:*)", "Bash(tail:*)",
               "Bash(wc:*)", "Bash(mkdir:*)", "Bash(cp:*)", "Bash(diff:*)"] + ([f"PowerShell({PY_S}:*)"] if WIN else [])
WEB = ["WebSearch", "WebFetch"]


def rtext(p):
    return Path(p).read_text(encoding="utf-8")


def wtext(p, s):
    Path(p).write_text(s, encoding="utf-8", newline="\n")


def child_env(agent=False):
    """子进程环境：UTF-8；BLAS 单线程（24 核混合架构上默认 24 线程反而慢 2 倍多，结果逐位不变）；
    不继承用户级的子智能体模型和推理强度设置（与原版运行保持默认一致，也省额度）。"""
    env = {k: v for k, v in os.environ.items() if k not in ("CLAUDE_CODE_SUBAGENT_MODEL", "CLAUDE_CODE_EFFORT_LEVEL")}
    env.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8", OPENBLAS_NUM_THREADS="1")
    if agent:
        env["PATH"] = f"{PY.parent}{os.pathsep}{env.get('PATH', '')}"
    return env


class BudgetExceeded(Exception):
    pass


class RemoteProgress(Exception):
    """GitHub 上出现了别处推送的新进度，按用户要求改按新进度继续。"""


class StopRequested(Exception):
    """runs/<运行>/STOP 存在：存档后停下。"""


class Run:
    def __init__(self, run_id, dry):
        self.dir = ROOT / "runs" / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.dry = dry
        self.lock = threading.RLock()
        self.status, self.usage, self.remote_flag = "running", None, False
        self.state_path = self.dir / "state.json"
        if self.state_path.exists():
            raw = rtext(self.state_path)
            old_root = json.loads(raw).get("root", "<ROOT>")
            esc = lambda s: json.dumps(s)[1:-1]  # noqa: E731
            raw = raw.replace(esc(old_root), esc(ROOT_S)).replace("<ROOT>", esc(ROOT_S))
            self.state = json.loads(raw)
        else:
            self.state = {"started": time.time(), "calls": 0, "cost_usd": 0.0, "steps": {}, "integrity": []}
        self.state["root"] = ROOT_S
        self.log_path = self.dir / "log.jsonl"
        self.common = rtext(PROMPTS / "_common.md")
        self.manifest = protocol_manifest()
        if "manifest" not in self.state:
            self.state["manifest"] = self.manifest
            self.save()
        elif self.state["manifest"] != self.manifest:
            raise SystemExit("评测协议文件与本次运行开始时不一致，拒绝继续")
        self.account_downtime()
        self.env = self.detect_env()
        if not dry:
            threading.Thread(target=self._watch, daemon=True).start()

    # ---------- 状态与日志 ----------
    def save(self):
        with self.lock:
            txt = gitsync.portable(json.dumps(self.state, indent=1, ensure_ascii=False))
            tmp = self.state_path.with_suffix(".tmp")
            wtext(tmp, txt)
            tmp.replace(self.state_path)

    def log(self, **kw):
        kw["t"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with self.lock, self.log_path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(gitsync.portable(json.dumps(kw, ensure_ascii=False)) + "\n")
        print(gitsync.portable(f"[{kw['t']}] {kw.get('event', '')} {kw.get('role', '')} {kw.get('msg', '')}"), flush=True)

    def step(self, name, fn):
        """已完成的步骤直接复用结果，保证断点续跑。"""
        if name in self.state["steps"]:
            return self.state["steps"][name]
        val = fn()
        self.state["steps"][name] = val
        self.save()
        self.checkpoint(f"完成 {name}")
        self.check_flags()
        return val

    def account_downtime(self):
        """上次停下（停机、暂停、关机）到这次启动之间的时间不计入 12 小时预算（原来要按 HANDOFF 手工补进 waited_s）。"""
        r = gitsync.read_runner(self.dir)
        last = r and r.get("last_alive")
        if last and time.time() - last > 120:
            gap = time.time() - last
            self.state["waited_s"] = self.state.get("waited_s", 0) + gap
            self.state.setdefault("downtime", []).append({"from": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(last)),
                                                          "to": time.strftime("%Y-%m-%d %H:%M:%S"), "secs": round(gap),
                                                          "host_before": r.get("host"), "host_now": gitsync.HOST})
            self.save()
            self.log(event="downtime", msg=f"上次心跳在 {gap / 3600:.1f} 小时前（{r.get('host')}，{r.get('status')}），这段时间不计入预算")
        if not self.dry:
            gitsync.write_runner(self.dir, self.status)

    def detect_env(self):
        """评测环境指纹。指纹不同，官方结果就不能直接比较（2026-09-28 从 Mac 换到 Windows 时发现，见 docs/JOURNAL.md）。
        2026-09-28 之前的所有结果（baseline/*.json、S4、S2）来自原来的 Mac，记为 "legacy"。"""
        r = subprocess.run([str(PY), str(ROOT / "tools" / "env_fingerprint.py")], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", env=child_env(), timeout=3600)
        if r.returncode != 0:
            raise SystemExit(f"算不出评测环境指纹：{r.stderr[-1000:]}")
        info = json.loads(r.stdout.strip().splitlines()[-1])
        envs = self.state.setdefault("envs", {"legacy": {"platform": "macOS arm64（Apple M4）",
                                                         "note": "2026-09-28 暂停前的全部官方结果：baseline/*.json、S4、S2"}})
        if info["fp"] not in envs:
            envs[info["fp"]] = {**info, "host": gitsync.HOST, "first_seen": time.strftime("%Y-%m-%d %H:%M:%S")}
            self.save()
        return info["fp"]

    def active_hours(self):
        hours = (time.time() - self.state["started"] - self.state.get("waited_s", 0)) / 3600
        return hours

    def check_budget(self):
        hours = self.active_hours()
        if hours > CFG["max_hours"] or self.state["calls"] >= CFG["max_calls"]:
            raise BudgetExceeded(f"预算用尽：{hours:.1f} 小时，{self.state['calls']} 次调用")

    def check_flags(self):
        if (self.dir / "STOP").exists():
            raise StopRequested()
        if self.remote_flag:
            raise RemoteProgress()

    # ---------- 存档、用量与同步 ----------
    def checkpoint(self, msg):
        if self.dry:
            return
        with self.lock:
            gitsync.write_runner(self.dir, self.status, env=self.env, usage=self.usage,
                                 budget_hours_used=round(self.active_hours(), 2), calls=self.state["calls"])
        res = gitsync.checkpoint(msg)
        print(f"[checkpoint] {msg} → {res}", flush=True)
        if res == "remote_ahead":
            self.remote_flag = True

    def _watch(self):
        """看门狗：每 3 分钟更新心跳、查用量；用量到 90%（7 天窗口 97%）时先存档一次；运行中每 30 分钟存档一次。"""
        last_save, last_hot = time.time(), 0.0
        while True:
            time.sleep(180)
            try:
                self.usage = gitsync.usage() or self.usage
                with self.lock:
                    gitsync.write_runner(self.dir, self.status, env=self.env, usage=self.usage,
                                         budget_hours_used=round(self.active_hours(), 2), calls=self.state["calls"])
                if self.status != "running":
                    continue
                fh, sd = (self.usage or {}).get("five_hour") or {}, (self.usage or {}).get("seven_day") or {}
                # 高用量存档 30 分钟内只做一次（重置时间每次查询有亚秒级抖动，不能拿它判断是不是同一个窗口）
                if (fh.get("pct", 0) >= 90 or sd.get("pct", 0) >= 97) and time.time() - last_hot > 1800:
                    last_hot = time.time()
                    self.checkpoint(f"用量已到 {fh.get('pct')}%（7 天窗口 {sd.get('pct')}%），额度用尽前先存档")
                    last_save = time.time()
                elif time.time() - last_save > 1800:
                    self.checkpoint("定时存档")
                    last_save = time.time()
            except Exception as e:  # noqa: BLE001 — 看门狗出错不能拖垮主流程
                print(f"[watchdog] {e!r}", flush=True)

    def wait_for_quota(self, text):
        """撞到订阅用量上限：存档推送 → 睡到重置（这段时间不计入预算）→ 先看 GitHub 上有没有新进度。"""
        secs = gitsync.reset_wait(text) + 180
        self.log(event="wait", msg=f"订阅用量上限，等待 {secs / 60:.0f} 分钟后重试：{text.strip()[:120]}")
        self.status = "waiting_for_quota"
        self.checkpoint(f"撞到订阅用量上限，预计 {time.strftime('%m-%d %H:%M', time.localtime(time.time() + secs))} 重置")
        end = time.time() + secs
        while time.time() < end:
            if (self.dir / "STOP").exists():
                break
            t0 = time.time()
            time.sleep(min(60, max(0.0, end - time.time())))
            with self.lock:
                self.state["waited_s"] = self.state.get("waited_s", 0) + (time.time() - t0)
        self.status = "running"
        self.save()
        self.check_flags()
        if gitsync.remote_news():
            self.log(event="remote", msg="额度恢复后查 GitHub：有别处推送的新进度，退出并按新进度继续")
            raise RemoteProgress()

    # ---------- 智能体调用 ----------
    def agent(self, role, cwd, model, tools, **vars_):
        self.check_budget()
        self.check_flags()
        body = rtext(PROMPTS / f"{role}.md")
        prompt = self.common + "\n\n" + body
        vars_ = {"ROOT": ROOT_S, "PY": PY_S, "WORKDIR": Path(cwd).as_posix(), "ENV": self.env,
                 "BASELINE_DIR": (ROOT / "baseline" / ("" if self.env == "legacy" else f"env-{self.env}")).as_posix().rstrip("/"),
                 "SHELL": "Git Bash on Windows" if WIN else "a POSIX shell", **vars_}
        for k, v in vars_.items():
            prompt = prompt.replace("{{" + k + "}}", v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, indent=1))
        left = re.findall(r"\{\{\w+\}\}", prompt)
        if left:
            raise ValueError(f"{role} 提示词有未填的占位符 {left}")
        Path(cwd).mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        self.state["calls"] += 1
        self.save()
        if self.dry:
            reply, meta = mock_agent(role, Path(cwd), vars_, self.state["calls"]), {"total_cost_usd": 0.0, "num_turns": 0}
        else:
            reply, meta = call_claude(prompt, cwd, model, tools, CFG["agent_timeout"], waiter=self.wait_for_quota)
        self.state["cost_usd"] += meta.get("total_cost_usd") or 0.0
        self.integrity_check(role)
        parsed = parse_json(reply)
        self.log(event="agent", role=role, model=model, cwd=Path(cwd).as_posix(), secs=round(time.time() - t0),
                 cost=meta.get("total_cost_usd"), turns=meta.get("num_turns"), session=meta.get("session_id"),
                 resumed=meta.get("resumed"), timed_out=meta.get("timed_out"), ok=parsed is not None,
                 msg=(json.dumps(parsed, ensure_ascii=False)[:300] if parsed else reply[-300:]))
        (self.dir / "transcripts").mkdir(exist_ok=True)
        wtext(self.dir / "transcripts" / f"{self.state['calls']:03d}_{role}.md",
              gitsync.portable(f"# {role} ({model})\n\n## Prompt\n\n{prompt}\n\n## Reply\n\n{reply}\n"))
        if parsed is None:
            raise RuntimeError(f"{role} 没有返回可解析的 JSON：{reply[-500:]}")
        return parsed

    def integrity_check(self, role):
        now = protocol_manifest()
        if now != self.state["manifest"]:
            changed = sorted(k for k in set(now) | set(self.state["manifest"]) if now.get(k) != self.state["manifest"].get(k))
            self.state["integrity"].append({"role": role, "changed": changed, "t": time.time()})
            self.save()
            raise SystemExit(f"完整性违规：{role} 之后评测协议文件被改动 {changed}")

    # ---------- 官方评测 ----------
    def evaluate(self, method, mode, out):
        """官方评测。结果文件里记下评测环境（eval_env）；已有结果只在同一环境下复用。"""
        out = Path(out)
        if out.exists():
            res = json.loads(rtext(out))
            env = res.get("eval_env", "legacy")
            if env == self.env or self.dry:
                return res
            stale = out.with_name(f"{out.stem}.env-{env}{out.suffix}")
            out.replace(stale)
            self.log(event="eval", msg=f"{out.name} 是在评测环境 {env} 下算的，改名为 {stale.name}，在本机环境 {self.env} 重跑")
        if self.dry:
            mode = "subset"  # 空跑只用子集，节省时间
        t0 = time.time()
        r = subprocess.run([str(PY), str(ROOT / "bench" / "harness.py"), "--method", str(method), "--mode", mode,
                            "--out", str(out)], capture_output=True, text=True, encoding="utf-8", errors="replace",
                           timeout=4 * 3600, env=child_env())
        wtext(out.with_suffix(".log"), gitsync.portable(r.stdout + r.stderr))
        if r.returncode != 0 or not out.exists():
            res = {"errors": {"harness": (r.stderr or r.stdout)[-3000:]}, "real": {}}
        else:
            res = json.loads(rtext(out))
        res["method"] = gitsync.portable(Path(res.get("method") or method).as_posix())
        res["eval_env"] = self.env
        wtext(out, json.dumps(res, indent=1, ensure_ascii=False))
        self.log(event="eval", msg=f"{mode} {Path(method).as_posix()} {time.time() - t0:.0f}s errors={list(res.get('errors', {}))}")
        return res

    def baseline(self, mode):
        """当前评测环境下的原版 TreeHFD 结果。legacy 环境用 baseline/*.json；其他环境第一次用时在本机跑一遍，
        存到 baseline/env-<指纹>/（原版的留出集指标本身带随机性，见 HANDOFF 已知问题，和原来一样只跑一次）。"""
        p = ROOT / "baseline" / (f"{mode}.json" if self.env == "legacy" else f"env-{self.env}/{mode}.json")
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            self.log(event="baseline", msg=f"评测环境 {self.env} 还没有基线，先在本机跑原版 TreeHFD（{mode}）")
            self.rebase_eval(ROOT / "bench" / "baseline_method.py", mode, p)
        return json.loads(rtext(p))

    def rebase_eval(self, method, mode, out):
        """换评测环境带来的重算（基线、旧结果）：这是换机器的额外开销，不是研究过程，用时不计入 12 小时预算，单独记账。"""
        t0 = time.time()
        res = self.evaluate(method, mode, out)
        dt = time.time() - t0
        with self.lock:
            self.state["waited_s"] = self.state.get("waited_s", 0) + dt
            self.state["rebaseline_s"] = self.state.get("rebaseline_s", 0) + dt
        self.save()
        self.log(event="rebase", msg=f"{Path(out).name} 重算用时 {dt / 60:.1f} 分钟，不计入预算")
        return res

    def full_result(self, trace):
        """某个想法在当前评测环境下的全量结果文件：原结果是别的环境算的，就在本机重跑同一份 method.py（结果另存）。"""
        if trace.get("env", "legacy") == self.env or self.dry:
            return trace["full_result"]
        ws = Path(trace["workspace"])
        out = ws / f"full_env-{self.env}.json"
        if not out.exists():
            orig = json.loads(rtext(trace["full_result"]))
            same = orig.get("method_sha256") == hashlib.sha256((ws / "method.py").read_bytes()).hexdigest()
            self.log(event="rebase", msg=f"{trace['idea']['id']} 的全量结果来自评测环境 {trace.get('env', 'legacy')}，"
                                         f"在本机环境 {self.env} 用同一份代码重跑（method.py 与原结果记录的哈希{'一致' if same else '不一致！'}）")
            self.rebase_eval(ws / "method.py", "full", out)
        return out.as_posix()


def rebase_refs(run):
    """换了评测环境时，把本次运行里已通过全量审查的想法在本机重跑（子集 + 全量），给之后的智能体和审查员当同环境参照
    （写在 prompts/_common.md 的执行环境说明里）。用时不计入预算。"""
    for name, t in list(run.state["steps"].items()):
        if name.startswith("idea_") and t.get("decision") == "good" and t.get("env", "legacy") != run.env:
            ws = Path(t["workspace"])
            out = ws / f"subset_env-{run.env}.json"
            if not out.exists():
                run.rebase_eval(ws / "method.py", "subset", out)
            run.full_result(t)


def protocol_manifest():
    files = sorted((ROOT / "bench").glob("*.py")) + sorted((ROOT / "bench").glob("*.md")) + \
        sorted((ROOT / "bench" / "data").glob("*.npz")) + sorted((ROOT / "tasks" / "treehfd" / "src" / "treehfd").glob("*.py"))
    return {f.relative_to(ROOT).as_posix(): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}


LIMIT_RE = re.compile(r"(hit your [\w ]{0,30}limit|usage limit|limit reached|rate.?limit|out of (?:extra )?usage)", re.I)
RESUME_PROMPT = ("Your previous turn was cut off (usage limit or a temporary error) before you finished. Continue the same task "
                 "from where you stopped (your partial work is in the working directory), then finish with the JSON reply your "
                 "role asks for.")
INTERRUPTED_NOTE = ("\n\nNote: a previous attempt at this exact task was cut off before it finished; its partial files may be in "
                    "the working directory. Review them, then complete or redo the task.")
WRAPUP_PROMPT = ("You have used up the time allowed for this task. Do not start anything new: make sure what you have is "
                 "consistent and runs, record the state in NOTES.md if your role keeps notes, then reply now with the JSON your "
                 "role asks for, stating honestly what is unfinished.")


def run_proc(cmd, stdin, cwd, timeout, env):
    """带超时地运行子进程；超时时连同它启动的子进程一起结束（Windows 上只杀父进程会留下智能体启动的自测进程，拖慢之后的官方计时）。"""
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd, env=env,
                         text=True, encoding="utf-8", errors="replace")
    try:
        out, err = p.communicate(stdin, timeout=timeout)
    except subprocess.TimeoutExpired:
        if WIN:
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(p.pid)], capture_output=True)
        p.kill()
        p.communicate()
        raise
    return subprocess.CompletedProcess(cmd, p.returncode, out, err)


def claude_bin():
    """claude 可执行文件。Windows 上 npm 装的是 claude.cmd 转发脚本，直接找它背后的 claude.exe（子进程调不了 .cmd）。"""
    exe = os.environ.get("CLAUDE_BIN") or shutil.which("claude") or "claude"
    if WIN and not exe.lower().endswith(".exe"):
        cand = Path(exe).parent / "node_modules" / "@anthropic-ai" / "claude-code" / "bin" / "claude.exe"
        if cand.exists():
            exe = str(cand)
    return exe


def call_claude(prompt, cwd, model, tools, timeout, waiter=None):
    """一次 `claude -p` 调用。提示词走标准输入（Windows 命令行最长 32767 个字符）。会话号事先指定，任何时候都能续接：
    - 撞到用量上限：waiter 负责存档和等待，之后用 --resume 接着原会话做；
    - 超过单次时限：原来直接算失败（整个步骤从写代码重来），现在续接一次、给 20 分钟收尾并如实交代没做完的部分，再超时才算失败；
    - 续接失败、API 临时错误、输出不是 JSON：最多再试 3 次；会话续不上就换新会话，带着"上次被打断"的说明重来。
    费用按所有尝试累加（原来被打断的那部分费用会丢；被强行结束的那次拿不到费用）。"""
    base = [claude_bin(), "-p", "--model", model, "--output-format", "json", "--allowedTools", *tools]
    sid = str(uuid.uuid4())
    cmd, stdin, resumed, limit = base + ["--session-id", sid], prompt, False, timeout
    cost, errors, waited, timeouts = 0.0, 0, 0.0, 0
    while True:
        try:
            r = run_proc(cmd, stdin, cwd, limit, child_env(agent=True))
        except subprocess.TimeoutExpired:
            timeouts += 1
            if timeouts > 1:
                return "智能体调用超时", {"total_cost_usd": cost, "session_id": sid}
            cmd, stdin, resumed, limit = base + ["--resume", sid], WRAPUP_PROMPT, True, min(1200, timeout)
            continue
        try:
            data = json.loads(r.stdout)
            text = data.get("result") or ""
        except json.JSONDecodeError:
            data, text = {}, (r.stdout + r.stderr)[-3000:]
        cost += data.get("total_cost_usd") or 0.0
        sid = data.get("session_id") or sid
        if LIMIT_RE.search(text) and len(text) < 400 and waited < 8 * 86400:
            t0 = time.time()
            if waiter:
                waiter(text)
            else:
                time.sleep((gitsync.seconds_until_reset(text) or 1800) + 180)
            waited += time.time() - t0
            cmd, stdin, resumed = base + ["--resume", sid], RESUME_PROMPT, True
            continue
        if data and not data.get("is_error"):
            data["total_cost_usd"] = cost
            data["resumed"] = resumed or timeouts > 0
            data["timed_out"] = timeouts > 0
            return text, data
        errors += 1
        if errors > 3:
            return text, {**data, "total_cost_usd": cost, "session_id": sid}
        time.sleep(30 * errors)
        if resumed and re.search(r"no conversation|not found", text, re.I):
            sid = str(uuid.uuid4())
            cmd, stdin, resumed = base + ["--session-id", sid], prompt + INTERRUPTED_NOTE, False
        else:
            cmd, stdin, resumed = base + ["--resume", sid], RESUME_PROMPT, True


def parse_json(text):
    blocks = re.findall(r"```json\s*(\{.*?\})\s*```", text or "", flags=re.S)
    cands = blocks[::-1] + [text[text.find("{"): text.rfind("}") + 1]] if text and "{" in text else blocks[::-1]
    for c in cands:
        try:
            return json.loads(c)
        except (json.JSONDecodeError, TypeError):
            continue
    return None


# ---------- 比较表：由脚本计算，审查员只读不算 ----------
AN_KEYS = ["mse_eta1", "mse_eta2", "mse_eta3", "mse_eta4", "mse_eta5", "mse_eta6", "mse_eta12", "mse_eta34",
           "mse_others", "resid_out", "fit_s"]
RD_KEYS = ["resid_in", "resid_out", "ortho_in", "ortho_out", "locvar_in", "fit_s"]


def fmt(v):
    return "NA" if v is None else (f"{v:.4g}" if isinstance(v, float) else str(v))


def rel(c, b):
    if c is None or b is None or b == 0:
        return ""
    return f"{(c - b) / abs(b) * 100:+.1f}%"


def compare_table(base, cands):
    """cands: [(名字, 结果)]。输出 Markdown 表：每列一个候选，附相对基线的变化。"""
    names = [n for n, _ in cands]
    lines = ["| metric | baseline | " + " | ".join(names) + " |", "|---" * (2 + len(names)) + "|"]
    ba = base.get("analytical", {}).get("mean", {})
    for k in AN_KEYS:
        row = [f"analytical.{k}", fmt(ba.get(k))]
        for _, c in cands:
            v = c.get("analytical", {}).get("mean", {}).get(k)
            row.append(f"{fmt(v)} ({rel(v, ba.get(k))})")
        lines.append("| " + " | ".join(row) + " |")
    for d in base.get("real", {}):
        for k in RD_KEYS:
            b = base["real"][d].get(k)
            row = [f"{d}.{k}", fmt(b)]
            for _, c in cands:
                v = c.get("real", {}).get(d, {}).get(k)
                row.append(f"{fmt(v)} ({rel(v, b)})")
            lines.append("| " + " | ".join(row) + " |")
    errs = {n: c.get("errors") for n, c in cands if c.get("errors")}
    if errs:
        lines.append(f"\nErrors reported by the harness: {json.dumps(errs, ensure_ascii=False)[:2000]}")
    return "\n".join(lines)


# ---------- 工作区 ----------
METHOD_TEMPLATE = '''"""Method under development (starts as a copy of the TreeHFD baseline)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))
from treehfd_mod import XGBTreeHFD  # noqa: E402


def fit(model, X_train, interaction_order=2):
    hfd = XGBTreeHFD(model)
    hfd.fit(X_train, interaction_order=interaction_order, verbose=False)
    return hfd


def predict(state, X):
    main, inter = state.predict(X, verbose=False)
    inter_list = state.interaction_list if inter.shape[1] else np.empty((0, 2), int)
    return {"intercept": state.eta0, "main": main, "inter": inter, "inter_list": inter_list}
'''


RESULT_FILES = ("subset_*.json", "full_*.json", "*selftest*.json", "audit_*.json", "*.log", ".git", ".wsgit",
                "HISTORY.bundle", "__pycache__")
OFFICIAL_RE = re.compile(r"(subset|full)_\d+\.json")


def new_workspace(path, from_ws=None):
    path = Path(path)
    if path.exists():
        return path
    if from_ws:
        shutil.copytree(from_ws, path, ignore=shutil.ignore_patterns(*RESULT_FILES, "ablations"))
    else:
        lib = path / "lib" / "treehfd_mod"
        shutil.copytree(ROOT / "tasks" / "treehfd" / "src" / "treehfd", lib, ignore=shutil.ignore_patterns("__pycache__"))
        for f in lib.glob("*.py"):
            f.chmod(0o644)
            wtext(f, rtext(f).replace("from treehfd.", "from treehfd_mod."))
        wtext(path / "method.py", METHOD_TEMPLATE)
    init_ws_git(path)
    commit(path, "workspace created")
    return path


def git(path, *args):
    """工作区自己的版本历史放在 <工作区>/.wsgit（不用嵌套 .git，外层仓库才能正常收录工作区文件）。"""
    path = Path(path)
    return subprocess.run(["git", f"--git-dir={path / '.wsgit'}", f"--work-tree={path}", *args],
                          cwd=path, capture_output=True, text=True, encoding="utf-8", errors="replace").stdout


def init_ws_git(path):
    git(path, "init", "-q")
    git(path, "config", "core.autocrlf", "false")
    excl = Path(path) / ".wsgit" / "info" / "exclude"
    excl.parent.mkdir(parents=True, exist_ok=True)
    wtext(excl, ".wsgit/\nHISTORY.bundle\n__pycache__/\n")


def commit(path, msg):
    gitsync.sanitize_tree(path)  # 智能体刚写完的文件里的本机路径 → 可移植写法（此时没有智能体在这个目录里工作）
    git(path, "add", "-A")
    git(path, "-c", "user.name=pipeline", "-c", "user.email=pipeline@local", "commit", "-q", "-m", msg, "--allow-empty")


# ---------- 各阶段 ----------
def stage_limitations(run):
    def go():
        lims, feedback = [], ""
        for r in range(CFG["lim_rounds"]):
            out = run.agent("limitation_extractor", run.dir / "work" / "limitations", FAST, READ,
                            EXISTING=lims or "(none)", FEEDBACK=feedback or "(none)")
            for x in out.get("limitations", []):
                x["id"] = f"L{len(lims) + 1}"
                lims.append(x)
            v = run.agent("limitation_verifier", run.dir / "work" / "limitations", FAST, READ, LIMITATIONS=lims)
            if v.get("decision") == "sufficient":
                break
            feedback = v.get("feedback", "")
        return lims
    return run.step("limitations", go)


def stage_seed_ideas(run, lims):
    def go():
        ideas = []
        while len(ideas) < CFG["n_seed"]:
            out = run.agent("idea_generator", run.dir / "work" / "ideas", STRONG, READ, N="2", LIMITATIONS=lims,
                            EXISTING=[{"id": i["id"], "title": i["title"]} for i in ideas] or "(none)")
            for idea in out.get("ideas", [])[: CFG["n_seed"] - len(ideas)]:
                idea["id"] = f"S{len(ideas) + 1}"
                nov = run.agent("novelty_checker", run.dir / "work" / "ideas", FAST, READ + WEB, IDEA=idea)
                idea["novelty"], idea["closest"] = nov.get("novelty", 0), nov.get("closest", [])
                ideas.append(idea)
        return sorted(ideas, key=lambda i: -i["novelty"])
    return run.step("seed_ideas", go)


def implement_idea(run, idea, base_sub, base_full):
    """论文 3.2 的 Idea Implementer：子集实现 → 子集审查/工程循环 → 全量验证 → 全量审查/工程循环。"""
    iid = idea["id"]
    ws = new_workspace(run.dir / "ideas" / iid)

    def go():
        trace = {"idea": idea, "workspace": ws.as_posix(), "env": run.env, "history": []}
        # 这个步骤从写代码重新开始，之前留下的官方评测结果（如果有）对应的是旧代码，不能复用
        for f in sorted(ws.iterdir()):
            if OFFICIAL_RE.fullmatch(f.name):
                stale = f.with_name(f"{f.stem}.stale-{time.strftime('%Y%m%d%H%M%S')}{f.suffix}")
                f.replace(stale)
                run.log(event="stale", msg=f"{iid}：{f.name} 属于上次没做完的尝试，改名为 {stale.name}")
        dirty = git(ws, "status", "--porcelain").strip()
        last = git(ws, "log", "-1", "--format=%s")
        dirty = dirty or any(w in last for w in ("interrupted", "stopped"))
        prompt_idea = idea
        if dirty:
            commit(ws, "partial work from an interrupted coder call")
            prompt_idea = {**idea, "note": "A previous implementation attempt of this idea was interrupted midway; its "
                           "partial files are in the working directory (see `git log`). Review them, then complete or redo."}
        run.agent("coder", ws, STRONG, CODE, IDEA=prompt_idea)
        commit(ws, "coder")
        for phase, base, critic in (("subset", base_sub, "subset_critic"), ("full", base_full, "fullset_critic")):
            for e in range(CFG["n_eng"] + 1):
                res = run.evaluate(ws / "method.py", phase, ws / f"{phase}_{e}.json")
                table = compare_table(base, [(iid, res)])
                v = run.agent(critic, ws, FAST, READ, IDEA=idea, TABLE=table)
                trace["history"].append({"phase": phase, "iter": e, "result": (ws / f"{phase}_{e}.json").as_posix(),
                                         "decision": v.get("decision"), "feedback": v.get("feedback")})
                if v.get("decision") in ("good", "bad"):
                    break
                if e == CFG["n_eng"]:
                    trace["history"][-1]["decision"] = "bad"
                    trace["history"][-1]["feedback"] = (trace["history"][-1]["feedback"] or "") + " [engineering budget exhausted → pruned]"
                    break
                run.agent("engineer", ws, STRONG, CODE, IDEA=idea, TABLE=table, FEEDBACK=v.get("feedback", ""), MODE=phase)
                commit(ws, f"engineer {phase} {e}")
            if trace["history"][-1]["decision"] != "good":
                break
        last = trace["history"][-1]
        trace["decision"] = "good" if last["phase"] == "full" and last["decision"] == "good" else "bad"
        trace["full_result"] = last["result"] if last["phase"] == "full" else None
        return trace
    return run.step(f"idea_{iid}", go)


def trace_digest(traces):
    return [{"id": t["idea"]["id"], "title": t["idea"]["title"], "mechanism": t["idea"].get("mechanism"),
             "decision": t["decision"], "history": [{k: h[k] for k in ("phase", "iter", "decision", "feedback")}
                                                    for h in t["history"]]} for t in traces]


def stage_idea_loop(run, seeds, base_sub, base_full):
    traces = []
    queue = list(seeds)
    for k in range(CFG["max_rounds"]):
        if k == 0:
            cands = [queue.pop(0) for _ in range(min(2, len(queue)))]
        else:
            evo = run.step(f"evolve_{k}", lambda: run.agent(
                "idea_evolver", run.dir / "work" / "ideas", STRONG, READ, TRACES=trace_digest(traces))["ideas"][0])
            evo["id"] = evo.get("id") or f"E{k}"
            cands = [evo] + ([queue.pop(0)] if queue else [])
        for c in cands:
            traces.append(implement_idea(run, c, base_sub, base_full))
            if sum(t["decision"] == "good" for t in traces) >= CFG["successes"]:
                return traces  # 攒够成功想法即提前终止（论文附录 A.2）
    return traces


def stage_select(run, good, base_full):
    if len(good) == 1:
        return good[0]["idea"]["id"]
    cands = [{"id": t["idea"]["id"], "title": t["idea"]["title"], "workspace": t["workspace"],
              "full_table": compare_table(base_full, [(t["idea"]["id"], json.loads(rtext(run.full_result(t))))])}
             for t in good]
    return run.step("select", lambda: run.agent("selector", run.dir / "work", FAST, READ, CANDIDATES=cands)["best"])


def stage_ablation(run, trace, base_full, tag):
    ws = Path(trace["workspace"])
    idea = trace["idea"]

    def go():
        plan = run.agent("ablation_planner", ws, FAST, READ, IDEA=idea)
        run.agent("ablation_coder", ws, STRONG, CODE, PLAN=plan)
        commit(ws, "ablations")
        full = json.loads(rtext(run.full_result(trace)))
        cols = [("full_method", full)]
        todo = [a["name"] for a in plan.get("ablations", []) if (ws / "ablations" / f"method_{a['name']}.py").exists()]
        # 分解是单线程的，消融变体并行评测以节省墙钟时间
        with ThreadPoolExecutor(max_workers=3) as pool:
            futs = {n: pool.submit(run.evaluate, ws / "ablations" / f"method_{n}.py", "full",
                                   ws / "ablations" / f"{n}_full.json") for n in todo}
            for n in todo:
                cols.append((f"w/o_{n}", futs[n].result()))
        table = compare_table(base_full, cols)
        wtext(ws / "ablation_table.md", table)
        v = run.agent("ablation_critic", ws, FAST, READ, IDEA=idea, TABLE=table)
        return {"plan": plan, "table": table, "verdict": v, "env": run.env}
    return run.step(f"ablation_{tag}", go)


def stage_ablation_refine(run, trace, base_full):
    """论文 3.4：消融审查要求改动时，工程智能体改出新版本，结果比较智能体判定更好才替换。"""
    abl = stage_ablation(run, trace, base_full, trace["idea"]["id"])
    for i in range(CFG["n_abl"]):
        if abl["verdict"].get("decision") != "refine":
            break
        new_id = f"{trace['idea']['id']}r{i + 1}"

        def go():
            ws2 = new_workspace(run.dir / "ideas" / new_id, from_ws=trace["workspace"])
            full = json.loads(rtext(run.full_result(trace)))
            run.agent("engineer", ws2, STRONG, CODE, IDEA=trace["idea"], TABLE=abl["table"],
                      FEEDBACK=abl["verdict"].get("feedback", ""), MODE="full")
            commit(ws2, "ablation-driven refinement")
            res2 = run.evaluate(ws2 / "method.py", "full", ws2 / "full_0.json")
            cmp = run.agent("result_comparator", run.dir / "work", FAST, READ,
                            A=compare_table(base_full, [("A", full)]), B=compare_table(base_full, [("B", res2)]))
            return {"ws": ws2.as_posix(), "result": (ws2 / "full_0.json").as_posix(), "prefer": cmp.get("prefer"), "env": run.env}
        r = run.step(f"ablation_refine_{new_id}", go)
        if r["prefer"] != "B":
            break
        trace = {**trace, "idea": {**trace["idea"], "id": new_id}, "workspace": r["ws"], "full_result": r["result"],
                 "env": r.get("env", "legacy")}
        abl = stage_ablation(run, trace, base_full, new_id)
    return trace, abl


def build_paper_dir(run, trace, abl, base_full, tag):
    pdir = run.dir / f"paper_{tag}"
    if not pdir.exists():
        ws = Path(trace["workspace"])
        shutil.copytree(ws, pdir / "method", ignore=shutil.ignore_patterns(*RESULT_FILES))
        (pdir / "results").mkdir(parents=True)
        shutil.copy(run.full_result(trace), pdir / "results" / "method_full.json")
        wtext(pdir / "results" / "baseline_full.json", json.dumps(base_full, indent=1, ensure_ascii=False))  # 与方法结果同一评测环境
        for f in (ws / "ablations").glob("*_full.json"):
            shutil.copy(f, pdir / "results" / f"ablation_{f.stem.replace('_full', '')}.json")
        wtext(pdir / "results" / "ablation_table.md", abl["table"])
        shutil.copy(ws / "NOTES.md", pdir / "NOTES.md") if (ws / "NOTES.md").exists() else None
        init_ws_git(pdir)
        commit(pdir, "paper workspace")
    return pdir


def stage_paper(run, trace, abl, base_full, tag):
    pdir = build_paper_dir(run, trace, abl, base_full, tag)

    def go():
        run.agent("drafter", pdir, FAST, CODE + WEB, )
        commit(pdir, "initial draft")
        reviews = []
        for r in range(CFG["n_peer"] + 1):
            rv = run.agent("reviewer", pdir, FAST, READ)
            reviews.append(rv)
            wtext(pdir / f"review_{r}.json", json.dumps(rv, indent=1, ensure_ascii=False))
            if rv.get("score", 0) >= CFG["review_threshold"] or r == CFG["n_peer"]:
                break
            plan = run.agent("rebuttal_planner", pdir, FAST, READ, REVIEW=rv)
            run.agent("rebuttal_coder", pdir, STRONG, CODE, TASKS=plan)
            run.agent("enhancer", pdir, STRONG, CODE, REVIEW=rv)
            commit(pdir, f"rebuttal round {r + 1}")
        meta = run.agent("meta_reviewer", pdir, FAST, READ, REVIEWS=reviews)
        return {"dir": pdir.as_posix(), "scores": [x.get("score") for x in reviews], "meta": meta}
    return run.step(f"paper_{tag}", go)


def stage_audit(run, paper, trace):
    pdir = Path(paper["dir"])

    def go():
        rerun = run.evaluate(pdir / "method" / "method.py", "full", pdir / "audit_rerun_full.json")
        orig = json.loads(rtext(run.full_result(trace)))
        drift = compare_table(orig, [("rerun", rerun)])
        wtext(pdir / "audit_rerun_vs_original.md", drift)
        report = {}
        for attempt in range(2):
            report["score"] = run.agent("claim_auditor", pdir, FAST, READ, RERUN=(pdir / "audit_rerun_full.json").as_posix())
            report["spec"] = run.agent("spec_auditor", pdir, FAST, READ, INTEGRITY=run.state["integrity"] or "no changes")
            report["refs"] = verify_refs(pdir)
            report["method_code"] = run.agent("method_code_auditor", pdir, FAST, READ)
            bad = (report["score"].get("mismatches") or report["spec"].get("violations") or report["refs"]["problems"]
                   or report["method_code"].get("discrepancies"))
            wtext(pdir / f"audit_{attempt}.json", json.dumps(report, indent=1, ensure_ascii=False))
            if not bad or attempt == 1:
                break
            run.agent("audit_fixer", pdir, STRONG, CODE, AUDIT=report)
            commit(pdir, "audit fixes")
        return {"final": report, "passed": not bad}
    return run.step("audit", go)


def verify_refs(pdir):
    refs = pdir / "refs.json"
    if not refs.exists():
        return {"problems": ["refs.json missing"], "summary": None}
    r = subprocess.run([sys.executable, str(VERIFY), str(refs)], capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=3600, env=child_env())
    try:
        out = json.loads(r.stdout)
    except json.JSONDecodeError:
        return {"problems": [f"verifier failed: {r.stderr[-500:]}"], "summary": None}
    return {"problems": [x for x in out["items"] if x["status"] not in ("id_ok", "title_only", "software_ok")],
            "summary": out["summary"]}


# ---------- 空跑用的假智能体 ----------
def mock_agent(role, cwd, v, n):
    j = lambda d: "```json\n" + json.dumps(d) + "\n```"  # noqa: E731
    if role == "limitation_extractor":
        return j({"limitations": [{"title": f"mock limitation {n}", "evidence": "x", "kind": "estimator", "why_actionable": "y"}]})
    if role == "limitation_verifier":
        return j({"decision": "insufficient" if n < 3 else "sufficient", "feedback": "need more"})
    if role in ("idea_generator", "idea_evolver"):
        return j({"ideas": [{"title": f"mock idea {n}-{k}", "addresses": [], "mechanism": "m", "implementation_sketch": "s",
                             "expected_effect_on_metrics": "e", "risks": "r"} for k in range(2)]})
    if role == "novelty_checker":
        return j({"novelty": n % 7 + 3, "closest": [], "rationale": "mock"})
    if role in ("coder", "engineer"):
        m = rtext(cwd / "method.py")
        wtext(cwd / "method.py", m + f"\n# mock edit {n}\n")
        wtext(cwd / "NOTES.md", "mock notes\n")
        return j({"status": "implemented", "summary": "mock", "selftest": "n/a", "changes": "mock"})
    if role == "subset_critic":
        return j({"decision": "refine" if "S" in str(cwd.name) and n % 2 else "good", "feedback": "mock feedback"})
    if role == "fullset_critic":
        return j({"decision": "good" if cwd.name != "S2" else "bad", "feedback": "mock"})
    if role == "selector":
        return j({"best": v["CANDIDATES"][0]["id"], "rationale": "mock"})
    if role == "ablation_planner":
        return j({"ablations": [{"name": "no_component_a", "removes": "a", "how": "baseline"}]})
    if role == "ablation_coder":
        (cwd / "ablations").mkdir(exist_ok=True)
        shutil.copy(cwd / "method.py", cwd / "ablations" / "method_no_component_a.py")
        src = rtext(cwd / "ablations" / "method_no_component_a.py").replace('parent / "lib"', 'parent.parent / "lib"')
        wtext(cwd / "ablations" / "method_no_component_a.py", src)
        return j({"status": "done", "files": [], "notes": ""})
    if role == "ablation_critic":
        return j({"decision": "refine", "feedback": "mock: drop component a"})
    if role == "result_comparator":
        return j({"prefer": "B", "rationale": "mock"})
    if role == "drafter":
        wtext(cwd / "paper.md", "# Mock paper\n\nResult: 0.0327\n")
        wtext(cwd / "refs.json", json.dumps([{"id": 1, "title": "Temporal difference learning of N-tuple networks for the game 2048",
                                              "authors": ["Marcin Szubert"], "year": 2014, "venue": "CIG",
                                              "doi": "10.1109/CIG.2014.6932907", "arxiv": None, "url": None}]))
        return j({"status": "done", "notes": ""})
    if role == "reviewer":
        return j({"summary": "s", "strengths": [], "weaknesses": ["w"], "questions": [], "score": 5 + 2 * (n % 2), "confidence": 3})
    if role == "rebuttal_planner":
        return j({"tasks": [{"id": "t1", "addresses": "w", "experiment": "e", "expected_output": "o"}]})
    if role in ("rebuttal_coder", "enhancer", "audit_fixer"):
        return j({"status": "done", "summary": "", "changes": ""})
    if role == "meta_reviewer":
        return j({"decision": "accept", "meta_review": "mock", "method_change": ""})
    if role == "claim_auditor":
        return j({"n_claims_checked": 1, "mismatches": []})
    if role == "spec_auditor":
        return j({"violations": [], "notes": ""})
    if role == "method_code_auditor":
        return j({"aligned": True, "discrepancies": []})
    raise KeyError(role)


def keep_awake():
    """Windows 上阻止系统空闲睡眠（相当于 macOS 的 caffeinate -i），进程退出后自动失效。"""
    if WIN:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED


# ---------- 主流程 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if WIN:
        sys.stdout.reconfigure(newline="\n")  # 输出日志统一 LF
    keep_awake()
    run = Run(a.run, a.dry_run)
    rc, status = 0, "stopped"
    run.log(event="start", msg=f"dry={a.dry_run} host={gitsync.HOST} env={run.env} cfg={CFG}")
    try:
        if a.dry_run:
            base_sub = json.loads(rtext(ROOT / "baseline" / "subset.json"))
            base_full = base_sub
        else:
            base_sub, base_full = run.baseline("subset"), run.baseline("full")
            rebase_refs(run)
        lims = stage_limitations(run)
        seeds = stage_seed_ideas(run, lims)
        traces = stage_idea_loop(run, seeds, base_sub, base_full)
        good = [t for t in traces if t["decision"] == "good"]
        if not good:
            run.log(event="stop", msg="没有想法在全量基准上超过基线，按论文 3.3 终止")
            status = "done"
            return
        best_id = stage_select(run, good, base_full)
        best = next(t for t in good if t["idea"]["id"] == best_id)
        best, abl = stage_ablation_refine(run, best, base_full)
        paper = stage_paper(run, best, abl, base_full, best["idea"]["id"])
        for m in range(CFG["n_meta"]):
            if paper["meta"].get("decision") != "refine":
                break
            tag = f"{best['idea']['id']}m{m + 1}"

            def meta_refine():
                ws2 = new_workspace(run.dir / "ideas" / tag, from_ws=best["workspace"])
                full = json.loads(rtext(run.full_result(best)))
                run.agent("engineer", ws2, STRONG, CODE, IDEA=best["idea"], TABLE=compare_table(base_full, [("current", full)]),
                          FEEDBACK=paper["meta"].get("method_change", ""), MODE="full")
                commit(ws2, "meta-review refinement")
                res2 = run.evaluate(ws2 / "method.py", "full", ws2 / "full_0.json")
                cmp = run.agent("result_comparator", run.dir / "work", FAST, READ,
                                A=compare_table(base_full, [("A", full)]), B=compare_table(base_full, [("B", res2)]))
                return {"ws": ws2.as_posix(), "result": (ws2 / "full_0.json").as_posix(), "prefer": cmp.get("prefer"), "env": run.env}
            r = run.step(f"meta_refine_{tag}", meta_refine)
            if r["prefer"] != "B":
                run.log(event="meta", msg="元审稿返工没有更好，保留原版本")
                break
            best = {**best, "idea": {**best["idea"], "id": tag}, "workspace": r["ws"], "full_result": r["result"],
                    "env": r.get("env", "legacy")}
            abl = stage_ablation(run, best, base_full, tag)
            paper = stage_paper(run, best, abl, base_full, tag)
        audit = stage_audit(run, paper, best)
        run.state["final"] = {"idea": best["idea"]["id"], "paper": paper, "audit_passed": audit["passed"], "env": run.env}
        run.save()
        run.log(event="done", msg=f"最终想法 {best['idea']['id']}，审稿分 {paper['scores']}，审计通过={audit['passed']}")
        status = "done"
    except BudgetExceeded as e:
        run.log(event="budget", msg=str(e))
        status = "budget_exhausted"
    except RemoteProgress:
        rc, status = gitsync.REMOTE_EXIT, "yielded_to_remote"
    except StopRequested:
        run.log(event="stop", msg="发现 STOP 文件，存档后停下")
        status = "stopped"
    except BaseException:
        status = "crashed"
        raise
    finally:
        run.status = status
        run.log(event="summary", msg=f"调用 {run.state['calls']} 次，折算成本 ${run.state['cost_usd']:.2f}，状态 {status}")
        if rc != gitsync.REMOTE_EXIT:  # 让位给 GitHub 上的新进度时不再推送本机状态（本机提交会被 autopilot 存到 backup 分支）
            run.checkpoint(f"编排器退出：{status}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
