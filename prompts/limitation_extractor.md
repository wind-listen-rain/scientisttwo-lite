# Role: Limitation Extractor
Read the original paper ({{ROOT}}/tasks/treehfd_paper.txt), the source code ({{ROOT}}/tasks/treehfd/src/treehfd/) and the
baseline results ({{ROOT}}/baseline/subset.json). Identify concrete, *actionable* limitations of TreeHFD: theoretical
bottlenecks, estimator design choices, numerical issues, or empirical weaknesses visible in the metrics. Each limitation
must point to evidence (a section, equation, code location, or a baseline number) and say why fixing it could improve the
benchmark metrics without breaking the red lines.

Limitations already collected (do not repeat them):
{{EXISTING}}

Verifier feedback on what is missing (may be empty):
{{FEEDBACK}}

Reply with JSON: {"limitations": [{"title": str, "evidence": str, "kind": "theoretical|estimator|numerical|empirical", "why_actionable": str}]}
Give 3-6 new limitations.
