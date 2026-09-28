"""核验一个 refs.json，把逐条结果和汇总以 JSON 打到标准输出。用法: python verify_refs.py <refs.json>"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import refcheck  # noqa: E402

refcheck.CACHE_FILE = Path(__file__).parent / "refcheck_cache.json"
refcheck.CACHE = json.loads(refcheck.CACHE_FILE.read_text()) if refcheck.CACHE_FILE.exists() else {}

refs = json.loads(Path(sys.argv[1]).read_text())
items = []
for r in refs:
    items.append(refcheck.check(r))
    refcheck.CACHE_FILE.write_text(json.dumps(refcheck.CACHE))
summary = {}
for x in items:
    summary[x["status"]] = summary.get(x["status"], 0) + 1
print(json.dumps({"summary": summary, "items": items}, ensure_ascii=False))
