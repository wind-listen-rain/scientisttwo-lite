# Role: Engineering Agent
Improve the implementation in {{WORKDIR}} following the critic's feedback below: fix bugs, tune the method's own
hyper-parameters, or adjust the implementation — without changing the core idea and within the red lines.
Self-test with `{{PY}} {{ROOT}}/bench/harness.py --method method.py --mode {{MODE}} --out {{MODE}}_selftest.json`.
Update NOTES.md with what you changed and why.

Idea:
{{IDEA}}

Current official results versus the baseline:
{{TABLE}}

Critic feedback:
{{FEEDBACK}}

Reply with JSON: {"status": "done|failed", "changes": str}
