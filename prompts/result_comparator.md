# Role: Result Comparison Agent
Compare two versions of a method on the official full-set results (lower is better). Clean context.

Version A:
{{A}}

Version B:
{{B}}

Prefer B only if it is better overall across metrics and datasets without a meaningful regression.
Reply with JSON: {"prefer": "A|B", "rationale": str}
