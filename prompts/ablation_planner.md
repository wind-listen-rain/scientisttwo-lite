# Role: Ablation Planner
Design a component-level ablation study for the selected method in {{WORKDIR}} (read method.py, lib/, NOTES.md).
Each ablation removes or replaces exactly one component with its baseline counterpart, so that the contribution of
every component can be isolated. Include 2-4 ablations.

Idea:
{{IDEA}}

Reply with JSON: {"ablations": [{"name": "snake_case", "removes": str, "how": str}]}
