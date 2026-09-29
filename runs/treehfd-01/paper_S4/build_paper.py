"""Fill paper_template.md with tables/*.md and tables/scalars.json -> paper.md.  Run make_tables.py first."""
import json, re
S = json.load(open("tables/scalars.json", encoding="utf-8"))
t = open("paper_template.md", encoding="utf-8").read()
def sub(m):
    k = m.group(1)
    if k.startswith("T:"): return open(f"tables/{k[2:]}.md", encoding="utf-8").read().strip()
    return str(S[k])
open("paper.md", "w", encoding="utf-8").write(re.sub(r"\{\{([^}]+)\}\}", sub, t))
