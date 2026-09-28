"""等某个步骤完成的那一刻让编排器停下，再自动重启 autopilot。改了编排器的代码或配置、又不想丢掉正在做的步骤时用。

用法: python tools/restart_after_step.py --run treehfd-01 --step idea_E1 --pid <正在运行的 autopilot 的进程号>
- 每秒看一次 runs/<运行>/state.json，这个步骤一出现在 steps 里就建 STOP 文件：编排器在该步骤存档之后、
  下一次智能体调用之前停下，已完成的工作都在 state.json 里；
- 等 autopilot 进程退出（它会先把最终状态推到 GitHub），再用磁盘上的新代码重启 autopilot（它启动时会删掉 STOP）；
- 如果 autopilot 在这个步骤完成之前就自己退出了（比如连续出错），就什么也不做，留给人来看。
动作都追加记在 runs/<运行>.out 里。
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WIN = os.name == "nt"


def alive(pid):
    if WIN:  # 注意：Windows 上 os.kill(pid, 0) 会直接结束那个进程，不能拿来探测
        import ctypes
        k = ctypes.windll.kernel32
        k.OpenProcess.restype = ctypes.c_void_p
        k.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        k.CloseHandle.argtypes = [ctypes.c_void_p]
        h = k.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
        if not h:
            return False
        still = k.WaitForSingleObject(h, 0) == 0x102  # WAIT_TIMEOUT：还在运行
        k.CloseHandle(h)
        return still
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--step", required=True)
    ap.add_argument("--pid", type=int, required=True)
    a = ap.parse_args()
    run_dir, out = ROOT / "runs" / a.run, ROOT / "runs" / f"{a.run}.out"

    def say(msg):
        with out.open("a", encoding="utf-8", newline="\n") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] restart_after_step {msg}\n")

    say(f"等步骤 {a.step} 完成后停下编排器并重启 autopilot（当前 autopilot 进程 {a.pid}）")
    while True:
        if not alive(a.pid):
            say(f"autopilot 在步骤 {a.step} 完成前就退出了，不自动重启，留给人来看")
            return 1
        try:
            if a.step in json.loads((run_dir / "state.json").read_text(encoding="utf-8"))["steps"]:
                break
        except (OSError, ValueError, KeyError):
            pass  # 编排器正在写 state.json，下一秒再读
        time.sleep(1)
    (run_dir / "STOP").touch()
    say(f"步骤 {a.step} 已完成，建了 STOP，等编排器存档停下、autopilot 推送后退出")
    while alive(a.pid):
        time.sleep(5)
    py = ROOT / ".conda" / ("python.exe" if WIN else "bin/python")
    kw = {"cwd": ROOT}
    if WIN:
        si = subprocess.STARTUPINFO()
        si.dwFlags, si.wShowWindow = subprocess.STARTF_USESHOWWINDOW, 7  # SW_SHOWMINNOACTIVE：最小化的新控制台窗口
        kw.update(creationflags=subprocess.CREATE_NEW_CONSOLE, startupinfo=si)
    else:
        kw.update(start_new_session=True)
    p = subprocess.Popen([str(py), "-u", str(ROOT / "tools" / "autopilot.py"), "--run", a.run], **kw)
    say(f"已用新代码重启 autopilot（进程 {p.pid}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
