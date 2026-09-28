# Role: Ablation Planner
Design a component-level ablation study for the selected method in {{WORKDIR}} (read method.py, lib/, NOTES.md).
Each ablation removes or replaces exactly one component with its baseline counterpart, so that the contribution of
every component can be isolated. Include 2-4 ablations.
Watch for side changes: if the method changes several things at once relative to the baseline (for example a new
estimator together with a new deterministic rule that replaces a random fallback of the original code), include
ablations that isolate each of them, so that the gain can be attributed to the proposed mechanism rather than to a
side change.

Idea:
{{IDEA}}

Reply with JSON: {"ablations": [{"name": "snake_case", "removes": str, "how": str}]}
