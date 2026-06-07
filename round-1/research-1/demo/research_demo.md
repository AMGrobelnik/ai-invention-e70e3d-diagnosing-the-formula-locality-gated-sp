# Tool Survey: FOL Provers, Structured LLM Prompting, OpenRouter Pricing, SBFL Libraries

## Summary

Four tool-selection questions are answered with confirmed APIs and code snippets:

1. **FOL Bounded Model Checking**: Z3 (`pip install z3-solver`) is the primary prover. Use `EnumSort` to create a finite N=8 domain; `solver.set(timeout=5000)` for 5s timeout; returns `sat`/`unsat`/`unknown` with explicit inconclusive verdict. NLTK/Mace4 (`end_size=8`) is the secondary cross-check but lacks explicit inconclusive verdict and requires an external binary (Docker-unfriendly).

2. **Span-Tagged FOL Prompting**: Code4Logic (NAACL 2025) uses Python-pseudocode basis functions but does NOT include span tags. NL2Logic also lacks span annotations. The executor must use a custom JSON prompt template (provided in research_report.md Section 2) that wraps Code4Logic's pseudocode style with `{"id", "span", "code"}` per component.

3. **OpenRouter Model Pricing**: Qwen2.5-72B-Instruct ($0.36/$0.40 per M tokens) is the primary model — best JSON structured output quality, total ~$0.20 for 150 sentences. Llama-3.3-70B-Instruct ($0.10/$0.32, free tier 200 req/day) is the fallback for Phase 0 pilots. The correct structured output parameter is `response_format: {"type": "json_schema", "json_schema": {"name": "...", "strict": true, "schema": {...}}}` — using `json_object` ignores the schema.

4. **SBFL Libraries**: Suresoft-GLaDOS/SBFL is pip-installable (`pip install git+https://github.com/Suresoft-GLaDOS/SBFL.git`) with OOP API `SBFL(formula='Ochiai').fit(X,y).ranks()`. agb94/sbfl requires `git clone + setup.py install` but provides functional API `ochiai(X, y)`. Both use rows=probes/cols=components matrix convention. Manual numpy fallback formulas are documented.

5. **Datasets**: FOLIO at `tasksource/folio` (HuggingFace, 1204 rows) has gold sentence-level FOL with nested quantifiers — suitable for the experiment. ProofWriter is primarily propositional and less suitable.

All findings are verified from primary sources (NLTK docs, Z3 tutorials, Code4Logic PDF Figure 9, OpenRouter docs, GitHub READMEs, HuggingFace).

## Research Findings

This survey covers four tool-selection questions for the spectrum-based FOL fault-localization experiment.

**FOL Bounded Model Checking**: Z3 (`pip install z3-solver`) is recommended as the primary prover [2, 3]. Using `EnumSort` to create a finite domain of N=8 constants makes FOL checking decidable; `solver.set(timeout=5000)` sets a 5-second timeout in milliseconds; and the solver returns an explicit three-way verdict: `sat` (countermodel found), `unsat` (entailment holds), or `unknown` (timeout) [3]. NLTK/Mace4 is the alternative [1]: `Mace(end_size=8).build_model(goal, assumptions)` searches for a countermodel up to domain size 8 (default 500), returning `True` if a countermodel is found and `False` otherwise — but NLTK/Mace4 provides no explicit `unknown` verdict to distinguish timeout from proven entailment. Mace4 also requires an external compiled binary which is problematic in Docker containers. Z3 is container-friendly and has a higher-quality Python API.

**Span-Tagged FOL Generation**: Code4Logic (NAACL 2025) [4] uses Python-pseudocode-style prompting with basis functions (Predicate, Conjunction, Implication, UniversalQuantification) to generate FOL bottom-up (confirmed from Figure 9 of the paper). However, Code4Logic does NOT include span tags — the components are not attributed to source sentence fragments. NL2Logic [5] similarly focuses on AST-guided syntactic accuracy (99%) without span annotations. The executor must therefore use a **custom prompt template** (designed in Section 2 of research_report.md) that wraps Code4Logic's pseudocode style with JSON output including `{"id", "span", "code"}` per component.

**OpenRouter Pricing**: Qwen2.5-72B-Instruct costs $0.36/M input and $0.40/M output [7], with explicit JSON structured output support. Llama-3.3-70B-Instruct costs $0.10/$0.32 per million tokens [6]. Total budget for 150 sentences is approximately $0.20 at Qwen and $0.09 at Llama — both well within the $10 limit. The OpenRouter structured output parameter is `response_format: {"type": "json_schema", "json_schema": {"name": "...", "strict": true, "schema": {...}}}` [8]. Using `json_object` instead of `json_schema` causes the model to ignore the schema.

**SBFL Libraries**: Suresoft-GLaDOS/SBFL [10] is pip-installable (`pip install git+https://github.com/Suresoft-GLaDOS/SBFL.git`) with OOP API: `SBFL(formula='Ochiai').fit(X, y).ranks(method='max')`. agb94/sbfl [9] is NOT on PyPI and requires `git clone + python setup.py install`. Both use rows=probes, cols=components matrix convention; X[i,j]=1 means probe i exercises component j; y[i]=0 means probe i fails.

**Datasets**: FOLIO is available at `tasksource/folio` on HuggingFace [11] with 1,204 rows containing gold whole-sentence FOL annotations, including nested quantifiers. ProofWriter [12] is available but primarily covers shallow propositional logic; FOLIO is preferred.

## Sources

[1] [NLTK Inference HOWTO](https://www.nltk.org/howto/inference.html) — Confirmed MaceCommand API with end_size parameter; Mace returns True (countermodel found) or False (no countermodel within end_size). Default end_size=500, can be set to 8 for bounded checking. No explicit inconclusive verdict.

[2] [Z3Py Tutorial — Advanced Examples](https://ericpony.github.io/z3py-tutorial/advanced-examples.htm) — Confirmed EnumSort API for finite domains and quantifier handling. Z3 may return unknown for undecidable/timeout cases.

[3] [CPMpy Z3 Solver Source](https://cpmpy.readthedocs.io/en/simplify_nested_wsum_boolexpr/_modules/cpmpy/solvers/z3.html) — Confirmed solver.set(timeout=ms) pattern; timeout is in milliseconds as integer. Bounded domain via explicit variable bound constraints.

[4] [Code4Logic: Code-Style Prompting for NL-to-FOL Translation (NAACL 2025)](https://aclanthology.org/2025.naacl-long.547.pdf) — Confirmed Python-pseudocode prompting with basis functions (Predicate, Conjunction, etc.). Figure 9 shows exact prompt template. No span tagging — components are not attributed to source sentence fragments.

[5] [NL2Logic: AST-Guided NL-to-FOL Translation](https://arxiv.org/abs/2602.13237) — AST-guided recursive LLM parser achieving 99% syntactic accuracy. No explicit span-to-component annotations.

[6] [OpenRouter — Llama-3.3-70B-Instruct](https://openrouter.ai/meta-llama/llama-3.3-70b-instruct) — Confirmed pricing: $0.10/M input, $0.32/M output.

[7] [OpenRouter — Qwen2.5-72B-Instruct](https://openrouter.ai/qwen/qwen-2.5-72b-instruct) — Confirmed pricing: $0.36/M input, $0.40/M output. Confirmed significant improvements in JSON/structured output generation.

[8] [OpenRouter Structured Outputs Documentation](https://openrouter.ai/docs/guides/features/structured-outputs) — Confirmed response_format parameter: type must be 'json_schema' (not 'json_object') for schema enforcement. Exact schema: {name, strict:true, schema:{...}}. Supports streaming with structured output.

[9] [agb94/sbfl — Spectrum-Based Fault Localization](https://github.com/agb94/sbfl) — NOT on PyPI. Requires git clone + setup.py install. API: ochiai(X, y) where X is coverage matrix (rows=tests, cols=elements), y is 0=fail/1=pass. Returns scores array.

[10] [Suresoft-GLaDOS/SBFL — SBFL Engine](https://github.com/Suresoft-GLaDOS/SBFL) — pip-installable via git URL. OOP API: SBFL(formula='Ochiai').fit(X,y).ranks(method='max'). Requires Python 3.9.1+. Boolean coverage matrix convention.

[11] [tasksource/folio — FOLIO Dataset](https://huggingface.co/datasets/tasksource/folio) — 1204 rows with gold FOL annotations. Columns: premises, premises-FOL, conclusion, conclusion-FOL, label. Contains nested quantifiers. Sentence-level FOL, not span-level.

[12] [tasksource/proofwriter — ProofWriter Dataset](https://huggingface.co/datasets/tasksource/proofwriter) — Available on HuggingFace. Primarily propositional/shallow Horn-clause logic. Less suitable than FOLIO for nested-quantifier FOL experiments.

## Follow-up Questions

- Does Z3's EnumSort bounded checking correctly handle equality axioms (e.g., Unique Name Assumption) needed for FOLIO-style multi-entity problems, or must the executor add explicit distinctness constraints?
- What is the empirical span-alignment accuracy of LLMs using the custom JSON prompt template versus automated post-hoc alignment (e.g., fuzzy matching component strings back to source words)?
- For the SBFL coverage matrix, should analytical probes be binary (component exercised y/n) or weighted by proof-path depth — and does Ochiai score quality degrade with very sparse matrices (few failing probes)?

---
*Generated by AI Inventor Pipeline*
