"""无人值守地跑完一次运行：同步 GitHub → 跑编排器 → 存档推送。撞到订阅用量上限时，编排器自己存档、等待、再续跑。

用法（在仓库根目录，用项目环境的 Python）:
  Windows: .conda\\python.exe -u tools\\autopilot.py --run treehfd-01
  macOS/Linux: .conda/bin/python -u tools/autopilot.py --run treehfd-01
停下：在 runs/<运行>/ 下建一个名为 STOP 的空文件，编排器会在下一次智能体调用前（或等待额度时）存档并退出。

每一轮：
1. 看 GitHub 上有没有新进度：有就按新进度来（能快进就快进；两边都有新提交时，本机提交推到 backup/ 分支后对齐 GitHub），
   并按 HISTORY.bundle 更新各工作区的版本历史；
2. GitHub 上的 RUNNER.json 显示另一台机器正在跑（45 分钟内有心跳）→ 不抢，每 15 分钟再看一次；
3. 核对评测协议 → 运行编排器（输出追加到 runs/<运行>.out）；
4. 编排器正常结束 → 推送后退出；退出码 75（GitHub 上有新进度）→ 回到第 1 步；
   其他异常 → 存档，5 分钟后重试，连续 3 次异常就停下等人来看。
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import gitsync  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    run_dir = ROOT / "runs" / a.run
    out = ROOT / "runs" / f"{a.run}.out"

    def say(msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] autopilot {msg}"
        print(line, flush=True)
        with out.open("a", encoding="utf-8", newline="\n") as f:
            f.write(gitsync.portable(line) + "\n")

    (run_dir / "STOP").unlink(missing_ok=True)  # 明确启动就表示要跑
    crashes = 0
    while True:
        news = gitsync.remote_news()
        if news:
            gitsync.adopt_remote(log=say)
        elif news is None:
            say("连不上 GitHub，先用本机进度继续")
        other = gitsync.other_runner_active(run_dir)
        if other:
            say(f"另一台机器 {other.get('host')} 正在跑（最后心跳 {other.get('last_alive_local')}），15 分钟后再看")
            time.sleep(900)
            continue
        chk = subprocess.run([sys.executable, str(ROOT / "tools" / "check_protocol.py"), str(run_dir / "state.json")],
                             capture_output=True, text=True, encoding="utf-8", errors="replace")
        if chk.returncode != 0:
            say(f"评测协议不一致，不续跑：{chk.stdout.strip()[-800:]}")
            return 1
        say(f"启动编排器（{gitsync.HOST}）")
        with out.open("a", encoding="utf-8", newline="\n") as f:
            p = subprocess.run([sys.executable, "-u", str(ROOT / "orchestrator.py"), "--run", a.run], cwd=ROOT,
                               stdout=f, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONUTF8": "1"})
        if p.returncode == 0:
            say("编排器正常结束（完成、预算用尽或按 STOP 停下），推送后退出")
            say(f"推送结果：{gitsync.checkpoint('autopilot 结束')}")
            return 0
        if p.returncode == gitsync.REMOTE_EXIT:
            say("编排器让位给 GitHub 上的新进度，同步后重启")
            crashes = 0
            continue
        crashes += 1
        say(f"编排器异常退出（返回码 {p.returncode}，第 {crashes} 次），错误见 {out.name}")
        gitsync.write_runner(run_dir, "crashed", returncode=p.returncode)
        say(f"推送结果：{gitsync.checkpoint(f'编排器异常退出（返回码 {p.returncode}）')}")
        if crashes >= 3:
            say("连续 3 次异常，停下等人来看")
            return 2
        time.sleep(300)


if __name__ == "__main__":
    sys.exit(main())
