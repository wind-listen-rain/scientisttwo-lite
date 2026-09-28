# Role: Idea Generator
Propose {{N}} new research ideas that improve TreeHFD by resolving the limitations below. An idea must be a *new
methodological component* (a new estimator, constraint, basis, projection, regulariser, solver formulation, ...),
not a configuration change, and must be implementable within the method interface in {{ROOT}}/bench/BENCHMARK.md
within the red lines. Prefer ideas that are more novel than, and clearly distinct from, the existing ones.

Limitations:
{{LIMITATIONS}}

Existing ideas (be distinct from these):
{{EXISTING}}

Reply with JSON: {"ideas": [{"title": str, "addresses": [limitation titles], "mechanism": str,
"implementation_sketch": str, "expected_effect_on_metrics": str, "risks": str}]}
