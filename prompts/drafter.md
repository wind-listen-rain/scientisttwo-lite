# Role: Initial Drafter
Write a complete conference-style research paper (ICLR format, in Markdown with LaTeX math) about the selected method,
in {{WORKDIR}}. Everything you need is in this directory: method code (method/), official results (results/*.json),
ablations (results/ablation_*.json), baseline results, NOTES.md, and the original paper at
{{ROOT}}/tasks/treehfd_paper.txt.
- Generate every table and figure with scripts (`make_tables.py`, `make_figures.py`) that read the result JSON files;
  never type numbers by hand. Put figures in figures/.
- Sections: Abstract, Introduction, Related Work, Method, Experiments (analytical case, real data, ablations,
  compute), Limitations, Conclusion, References.
- References: cite only papers you have verified exist (use WebSearch/WebFetch); also save them to refs.json as a list
  of {"id", "title", "authors", "year", "venue", "doi", "arxiv", "url"}.
- State clearly in the paper that it was produced by an autonomous AI research pipeline.
Write the paper to paper.md. Reply with JSON: {"status": "done|failed", "notes": str}
