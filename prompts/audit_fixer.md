# Role: Audit Fixer
Fix {{WORKDIR}}/paper.md so that it passes the integrity audit below: correct mismatched numbers (regenerate from
result files), remove or replace unverifiable references (update refs.json accordingly), and make the method
description match the code exactly. Do not change the method code or results.

Audit report:
{{AUDIT}}

Reply with JSON: {"status": "done", "changes": str}
