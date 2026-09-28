"""核对本机的评测协议文件是否与某次运行开始时记录的一致（续跑前必查）。

用法: python tools/check_protocol.py [runs/treehfd-01/state.json]
不一致时编排器会拒绝续跑；这里逐个列出差异，方便定位是数据、评测脚本还是 treehfd 源码版本的问题。
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    state = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "runs" / "treehfd-01" / "state.json"
    import importlib.util
    spec = importlib.util.spec_from_file_location("orchestrator", ROOT / "orchestrator.py")
    orch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(orch)
    want = json.loads(state.read_text())["manifest"]
    have = orch.protocol_manifest()
    bad = sorted(k for k in set(want) | set(have) if want.get(k) != have.get(k))
    if not bad:
        print(f"评测协议一致：{len(want)} 个文件的 SHA-256 全部匹配，可以续跑")
        return
    print(f"评测协议不一致（{len(bad)} 个文件）：")
    for k in bad:
        print(f"  {k}: 记录={str(want.get(k))[:12]}  本机={str(have.get(k))[:12]}")
    sys.exit(1)


if __name__ == "__main__":
    main()
