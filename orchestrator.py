"""ScientistTwo-lite：按 ScientistTwo（arXiv 2609.19644）第 3 节与附录 A.2 复现的简化版自主科研流水线。

用法:
  python orchestrator.py --run <run_id> [--dry-run]
每个角色是一次独立的 `claude -p` 调用（提示词在 prompts/），审查员总在干净上下文里运行。
官方评测由本脚本调用只读的 bench/harness.py 完成，智能体自测的结果不作数。
所有中间状态存于 runs/<run_id>/state.json，中断后重跑同一命令会从断点继续。
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = ROOT / ".conda" / "bin" / "python"
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
    "max_hours": 12,          # 全局墙钟上限
    "max_calls": 110,         # 全局智能体调用上限
    "agent_timeout": 3600,    # 单次智能体调用超时秒数
}
FAST, STRONG = "sonnet", "opus"  # 对应原文 Gemini 3.6 Flash / Claude Code + Opus 4.8
READ = ["Read", "Glob", "Grep"]
CODE = READ + ["Write", "Edit", f"Bash({PY}:*)", "Bash(python:*)", "Bash(ls:*)", "Bash(cat:*)", "Bash(head:*)", "Bash(tail:*)",
               "Bash(wc:*)", "Bash(mkdir:*)", "Bash(cp:*)", "Bash(diff:*)"]
WEB = ["WebSearch", "WebFetch"]


class BudgetExceeded(Exception):
    pass


class Run:
    def __init__(self, run_id, dry):
        self.dir = ROOT / "runs" / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.dry = dry
        self.state_path = self.dir / "state.json"
        if self.state_path.exists():
            raw = self.state_path.read_text()
            old_root = json.loads(raw).get("root", "<ROOT>")
            raw = raw.replace(json.dumps(old_root)[1:-1], json.dumps(str(ROOT))[1:-1]).replace("<ROOT>", str(ROOT))
            self.state = json.loads(raw)
        else:
            self.state = {"started": time.time(), "calls": 0, "cost_usd": 0.0, "steps": {}, "integrity": []}
        self.state["root"] = str(ROOT)
        self.log_path = self.dir / "log.jsonl"
        self.common = (PROMPTS / "_common.md").read_text()
        self.manifest = protocol_manifest()
        if "manifest" not in self.state:
            self.state["manifest"] = self.manifest
            self.save()
        elif self.state["manifest"] != self.manifest:
            raise SystemExit("评测协议文件与本次运行开始时不一致，拒绝继续")

    # ---------- 状态与日志 ----------
    def save(self):
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1, ensure_ascii=False))
        tmp.replace(self.state_path)

    def log(self, **kw):
        kw["t"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with self.log_path.open("a") as f:
            f.write(json.dumps(kw, ensure_ascii=False) + "\n")
        print(f"[{kw['t']}] {kw.get('event', '')} {kw.get('role', '')} {kw.get('msg', '')}", flush=True)

    def step(self, name, fn):
        """已完成的步骤直接复用结果，保证断点续跑。"""
        if name in self.state["steps"]:
            return self.state["steps"][name]
        val = fn()
        self.state["steps"][name] = val
        self.save()
        return val

    def on_wait(self, secs, text):
        self.state["waited_s"] = self.state.get("waited_s", 0) + secs
        self.save()
        self.log(event="wait", msg=f"订阅用量上限，等待 {secs / 60:.0f} 分钟后重试：{text.strip()[:120]}")

    def check_budget(self):
        hours = (time.time() - self.state["started"] - self.state.get("waited_s", 0)) / 3600
        if hours > CFG["max_hours"] or self.state["calls"] >= CFG["max_calls"]:
            raise BudgetExceeded(f"预算用尽：{hours:.1f} 小时，{self.state['calls']} 次调用")

    # ---------- 智能体调用 ----------
    def agent(self, role, cwd, model, tools, **vars_):
        self.check_budget()
        body = (PROMPTS / f"{role}.md").read_text()
        prompt = self.common + "\n\n" + body
        vars_ = {"ROOT": str(ROOT), "PY": str(PY), "WORKDIR": str(cwd), **vars_}
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
            reply, meta = call_claude(prompt, cwd, model, tools, CFG["agent_timeout"], on_wait=self.on_wait)
        self.state["cost_usd"] += meta.get("total_cost_usd") or 0.0
        self.integrity_check(role)
        parsed = parse_json(reply)
        self.log(event="agent", role=role, model=model, cwd=str(cwd), secs=round(time.time() - t0),
                 cost=meta.get("total_cost_usd"), turns=meta.get("num_turns"), session=meta.get("session_id"),
                 ok=parsed is not None, msg=(json.dumps(parsed, ensure_ascii=False)[:300] if parsed else reply[-300:]))
        (self.dir / "transcripts").mkdir(exist_ok=True)
        (self.dir / "transcripts" / f"{self.state['calls']:03d}_{role}.md").write_text(
            f"# {role} ({model})\n\n## Prompt\n\n{prompt}\n\n## Reply\n\n{reply}\n")
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
        out = Path(out)
        if out.exists():
            return json.loads(out.read_text())
        if self.dry:
            mode = "subset"  # 空跑只用子集，节省时间
        t0 = time.time()
        r = subprocess.run([str(PY), str(ROOT / "bench" / "harness.py"), "--method", str(method), "--mode", mode,
                            "--out", str(out)], capture_output=True, text=True, timeout=4 * 3600)
        (out.with_suffix(".log")).write_text(r.stdout + r.stderr)
        if r.returncode != 0 or not out.exists():
            res = {"errors": {"harness": (r.stderr or r.stdout)[-3000:]}, "real": {}}
            out.write_text(json.dumps(res))
        res = json.loads(out.read_text())
        self.log(event="eval", msg=f"{mode} {method} {time.time() - t0:.0f}s errors={list(res.get('errors', {}))}")
        return res


def protocol_manifest():
    files = sorted((ROOT / "bench").glob("*.py")) + sorted((ROOT / "bench").glob("*.md")) + \
        sorted((ROOT / "bench" / "data").glob("*.npz")) + sorted((ROOT / "tasks" / "treehfd" / "src" / "treehfd").glob("*.py"))
    return {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}


LIMIT_RE = re.compile(r"(hit your \w+ limit|usage limit|limit reached|rate.?limit)", re.I)


def seconds_until_reset(text):
    """从 "resets 10:40pm" 这类提示里算出还要等多少秒；解析不了返回 None。"""
    m = re.search(r"resets\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)", text or "", re.I)
    if not m:
        return None
    h, mi = int(m.group(1)) % 12 + (12 if m.group(3).lower() == "pm" else 0), int(m.group(2) or 0)
    now = time.localtime()
    target = time.mktime((now.tm_year, now.tm_mon, now.tm_mday, h, mi, 0, 0, 0, -1))
    if target <= time.time():
        target += 86400
    return target - time.time()


def call_claude(prompt, cwd, model, tools, timeout, on_wait=None):
    env = os.environ.copy()
    env["PATH"] = f"{PY.parent}:{env['PATH']}"
    cmd = ["claude", "-p", prompt, "--model", model, "--output-format", "json", "--allowedTools", *tools]
    bad_json = 0
    waited = 0
    while True:
        try:
            r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, env=env)
        except subprocess.TimeoutExpired:
            return "智能体调用超时", {}
        try:
            data = json.loads(r.stdout)
            text = data.get("result") or ""
        except json.JSONDecodeError:
            data, text = {}, (r.stdout + r.stderr)[-3000:]
        if LIMIT_RE.search(text) and len(text) < 400 and waited < 24 * 3600:
            wait = (seconds_until_reset(text) or 1800) + 180
            if on_wait:
                on_wait(wait, text)
            time.sleep(wait)
            waited += wait
            continue
        if data:
            return text, data
        bad_json += 1
        if bad_json >= 2:
            return text, {}
        time.sleep(30)


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
            f.write_text(f.read_text().replace("from treehfd.", "from treehfd_mod."))
        (path / "method.py").write_text(METHOD_TEMPLATE)
    init_ws_git(path)
    commit(path, "workspace created")
    return path


def git(path, *args):
    """工作区自己的版本历史放在 <工作区>/.wsgit（不用嵌套 .git，外层仓库才能正常收录工作区文件）。"""
    path = Path(path)
    return subprocess.run(["git", f"--git-dir={path / '.wsgit'}", f"--work-tree={path}", *args],
                          cwd=path, capture_output=True, text=True).stdout


def init_ws_git(path):
    git(path, "init", "-q")
    excl = Path(path) / ".wsgit" / "info" / "exclude"
    excl.parent.mkdir(parents=True, exist_ok=True)
    excl.write_text(".wsgit/\nHISTORY.bundle\n__pycache__/\n")


def commit(path, msg):
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
        trace = {"idea": idea, "workspace": str(ws), "history": []}
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
                trace["history"].append({"phase": phase, "iter": e, "result": str(ws / f"{phase}_{e}.json"),
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
              "full_table": compare_table(base_full, [(t["idea"]["id"], json.loads(Path(t["full_result"]).read_text()))])}
             for t in good]
    return run.step("select", lambda: run.agent("selector", run.dir / "work", FAST, READ, CANDIDATES=cands)["best"])


def stage_ablation(run, trace, base_full, tag):
    ws = Path(trace["workspace"])
    idea = trace["idea"]

    def go():
        plan = run.agent("ablation_planner", ws, FAST, READ, IDEA=idea)
        run.agent("ablation_coder", ws, STRONG, CODE, PLAN=plan)
        commit(ws, "ablations")
        full = json.loads(Path(trace["full_result"]).read_text())
        cols = [("full_method", full)]
        todo = [a["name"] for a in plan.get("ablations", []) if (ws / "ablations" / f"method_{a['name']}.py").exists()]
        # 分解是单线程的，消融变体并行评测以节省墙钟时间
        with ThreadPoolExecutor(max_workers=3) as pool:
            futs = {n: pool.submit(run.evaluate, ws / "ablations" / f"method_{n}.py", "full",
                                   ws / "ablations" / f"{n}_full.json") for n in todo}
            for n in todo:
                cols.append((f"w/o_{n}", futs[n].result()))
        table = compare_table(base_full, cols)
        (ws / "ablation_table.md").write_text(table)
        v = run.agent("ablation_critic", ws, FAST, READ, IDEA=idea, TABLE=table)
        return {"plan": plan, "table": table, "verdict": v}
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
            full = json.loads(Path(trace["full_result"]).read_text())
            run.agent("engineer", ws2, STRONG, CODE, IDEA=trace["idea"], TABLE=abl["table"],
                      FEEDBACK=abl["verdict"].get("feedback", ""), MODE="full")
            commit(ws2, "ablation-driven refinement")
            res2 = run.evaluate(ws2 / "method.py", "full", ws2 / "full_0.json")
            cmp = run.agent("result_comparator", run.dir / "work", FAST, READ,
                            A=compare_table(base_full, [("A", full)]), B=compare_table(base_full, [("B", res2)]))
            return {"ws": str(ws2), "result": str(ws2 / "full_0.json"), "prefer": cmp.get("prefer")}
        r = run.step(f"ablation_refine_{new_id}", go)
        if r["prefer"] != "B":
            break
        trace = {**trace, "idea": {**trace["idea"], "id": new_id}, "workspace": r["ws"], "full_result": r["result"]}
        abl = stage_ablation(run, trace, base_full, new_id)
    return trace, abl


def build_paper_dir(run, trace, abl, base_full, tag):
    pdir = run.dir / f"paper_{tag}"
    if not pdir.exists():
        ws = Path(trace["workspace"])
        shutil.copytree(ws, pdir / "method", ignore=shutil.ignore_patterns(*RESULT_FILES))
        (pdir / "results").mkdir(parents=True)
        shutil.copy(trace["full_result"], pdir / "results" / "method_full.json")
        shutil.copy(ROOT / "baseline" / "full.json", pdir / "results" / "baseline_full.json")
        for f in (ws / "ablations").glob("*_full.json"):
            shutil.copy(f, pdir / "results" / f"ablation_{f.stem.replace('_full', '')}.json")
        (pdir / "results" / "ablation_table.md").write_text(abl["table"])
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
            (pdir / f"review_{r}.json").write_text(json.dumps(rv, indent=1, ensure_ascii=False))
            if rv.get("score", 0) >= CFG["review_threshold"] or r == CFG["n_peer"]:
                break
            plan = run.agent("rebuttal_planner", pdir, FAST, READ, REVIEW=rv)
            run.agent("rebuttal_coder", pdir, STRONG, CODE, TASKS=plan)
            run.agent("enhancer", pdir, STRONG, CODE, REVIEW=rv)
            commit(pdir, f"rebuttal round {r + 1}")
        meta = run.agent("meta_reviewer", pdir, FAST, READ, REVIEWS=reviews)
        return {"dir": str(pdir), "scores": [x.get("score") for x in reviews], "meta": meta}
    return run.step(f"paper_{tag}", go)


def stage_audit(run, paper, trace):
    pdir = Path(paper["dir"])

    def go():
        rerun = run.evaluate(pdir / "method" / "method.py", "full", pdir / "audit_rerun_full.json")
        orig = json.loads(Path(trace["full_result"]).read_text())
        drift = compare_table(orig, [("rerun", rerun)])
        (pdir / "audit_rerun_vs_original.md").write_text(drift)
        report = {}
        for attempt in range(2):
            report["score"] = run.agent("claim_auditor", pdir, FAST, READ, RERUN=str(pdir / "audit_rerun_full.json"))
            report["spec"] = run.agent("spec_auditor", pdir, FAST, READ, INTEGRITY=run.state["integrity"] or "no changes")
            report["refs"] = verify_refs(pdir)
            report["method_code"] = run.agent("method_code_auditor", pdir, FAST, READ)
            bad = (report["score"].get("mismatches") or report["spec"].get("violations") or report["refs"]["problems"]
                   or report["method_code"].get("discrepancies"))
            (pdir / f"audit_{attempt}.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
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
    r = subprocess.run([sys.executable, str(VERIFY), str(refs)], capture_output=True, text=True, timeout=3600)
    try:
        out = json.loads(r.stdout)
    except json.JSONDecodeError:
        return {"problems": [f"verifier failed: {r.stderr[-500:]}"], "summary": None}
    return {"problems": [x for x in out["items"] if x["status"] not in ("id_ok", "title_only", "software_ok")],
            "summary": out["summary"]}


# ---------- 空跑用的假智能体 ----------
def mock_agent(role, cwd, v, n):
    j = lambda d: "```json\n" + json.dumps(d) + "\n```"
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
        m = (cwd / "method.py").read_text()
        (cwd / "method.py").write_text(m + f"\n# mock edit {n}\n")
        (cwd / "NOTES.md").write_text("mock notes\n")
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
        src = (cwd / "ablations" / "method_no_component_a.py").read_text().replace('parent / "lib"', 'parent.parent / "lib"')
        (cwd / "ablations" / "method_no_component_a.py").write_text(src)
        return j({"status": "done", "files": [], "notes": ""})
    if role == "ablation_critic":
        return j({"decision": "refine", "feedback": "mock: drop component a"})
    if role == "result_comparator":
        return j({"prefer": "B", "rationale": "mock"})
    if role == "drafter":
        (cwd / "paper.md").write_text("# Mock paper\n\nResult: 0.0327\n")
        (cwd / "refs.json").write_text(json.dumps([{"id": 1, "title": "Temporal difference learning of N-tuple networks for the game 2048",
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


# ---------- 主流程 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    run = Run(a.run, a.dry_run)
    base_sub = json.loads((ROOT / "baseline" / "subset.json").read_text())
    base_full = json.loads((ROOT / "baseline" / ("subset.json" if a.dry_run else "full.json")).read_text())
    run.log(event="start", msg=f"dry={a.dry_run} cfg={CFG}")
    try:
        lims = stage_limitations(run)
        seeds = stage_seed_ideas(run, lims)
        traces = stage_idea_loop(run, seeds, base_sub, base_full)
        good = [t for t in traces if t["decision"] == "good"]
        if not good:
            run.log(event="stop", msg="没有想法在全量基准上超过基线，按论文 3.3 终止")
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
                full = json.loads(Path(best["full_result"]).read_text())
                run.agent("engineer", ws2, STRONG, CODE, IDEA=best["idea"], TABLE=compare_table(base_full, [("current", full)]),
                          FEEDBACK=paper["meta"].get("method_change", ""), MODE="full")
                commit(ws2, "meta-review refinement")
                res2 = run.evaluate(ws2 / "method.py", "full", ws2 / "full_0.json")
                cmp = run.agent("result_comparator", run.dir / "work", FAST, READ,
                                A=compare_table(base_full, [("A", full)]), B=compare_table(base_full, [("B", res2)]))
                return {"ws": str(ws2), "result": str(ws2 / "full_0.json"), "prefer": cmp.get("prefer")}
            r = run.step(f"meta_refine_{tag}", meta_refine)
            if r["prefer"] != "B":
                run.log(event="meta", msg="元审稿返工没有更好，保留原版本")
                break
            best = {**best, "idea": {**best["idea"], "id": tag}, "workspace": r["ws"], "full_result": r["result"]}
            abl = stage_ablation(run, best, base_full, tag)
            paper = stage_paper(run, best, abl, base_full, tag)
        audit = stage_audit(run, paper, best)
        run.state["final"] = {"idea": best["idea"]["id"], "paper": paper, "audit_passed": audit["passed"]}
        run.save()
        run.log(event="done", msg=f"最终想法 {best['idea']['id']}，审稿分 {paper['scores']}，审计通过={audit['passed']}")
    except BudgetExceeded as e:
        run.log(event="budget", msg=str(e))
    finally:
        run.log(event="summary", msg=f"调用 {run.state['calls']} 次，折算成本 ${run.state['cost_usd']:.2f}")


if __name__ == "__main__":
    main()
