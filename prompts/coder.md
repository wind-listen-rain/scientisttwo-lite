# Role: Idea Coding Agent
Implement the research idea below as a method in your working directory {{WORKDIR}}.
- `method.py` already contains a working copy of the baseline; `lib/treehfd_mod/` is a modifiable copy of the original
  package (imported by method.py). Change whatever you need inside {{WORKDIR}}.
- Keep the interface in {{ROOT}}/bench/BENCHMARK.md exactly.
- Self-test with the subset benchmark: `{{PY}} {{ROOT}}/bench/harness.py --method method.py --mode subset --out subset_selftest.json`
  and iterate until the method runs without errors. The official evaluation is re-run by the pipeline afterwards.
- Write NOTES.md: what you implemented, every design decision and hyper-parameter, and your self-test numbers.

Idea:
{{IDEA}}

Reply with JSON: {"status": "implemented|failed", "summary": str, "selftest": str}
