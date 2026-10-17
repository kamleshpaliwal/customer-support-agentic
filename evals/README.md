# Evaluation dataset

## Install evaluation dependencies

Ragas is an optional dependency, so the customer-support application does not need it at runtime. From the project root, activate the virtual environment and install the project with the `eval` extra:

```bash
python -m pip install -e ".[eval]"
```

This installs Ragas for the evaluation tooling while leaving the default application install unchanged. The baseline runner below does not call Ragas yet.

## Run the baseline evaluator

The runner invokes the configured LangGraph workflow for each selected case and saves a JSON report under `outputs/evaluations/`. It records the actual route, MCP tool and arguments, clarification or elicitation prompt, specialist evidence, and final answer. Summary checks cover intent routing, tool selection, the final tool invocation's arguments, expected action, exact fee values, and latency percentiles. Academic evidence and answers are saved for inspection, but are not given an automated quality score yet; Ragas metrics are a later step.

Before running academic or finance cases, start Ollama, initialize the database, and make sure the academic PDFs have been ingested. The evaluator uses the providers configured in `.env`; selecting Bedrock can make billable AWS model calls. Begin with a single case:

```bash
python evals/run_eval.py --case-id FIN-001
```

Run the full dataset after checking the first report:

```bash
python evals/run_eval.py
```

You can choose an output path or run a small subset:

```bash
python evals/run_eval.py --limit 3 --output outputs/evaluations/first-three.json
```

For MCP language-elicitation cases, the runner records the pending question and options. It cancels the request afterward to avoid choosing a language on the user's behalf. A dataset case explicitly labeled as a user cancellation is simulated as cancelled.

`datasets/support_eval.jsonl` is the initial human-labeled evaluation set for the customer-support agent. It contains 27 cases covering fee lookups, missing or invalid lookup fields, a similar-board confirmation, unsupported requests, syllabus retrieval, book recommendations, source limitations, and MCP language elicitation.

## Dataset fields

- `id`, `category`, `user_input`: stable test-case identity, area, and query.
- `expected_intent`, `expected_action`: expected supervisor route and high-level behavior.
- `expected_tool`, `expected_tool_arguments`: expected tool choice and semantic argument constraints. These are labels for evaluation; they do not claim that the current tool API accepts each value as a separate argument.
- `expected_clarification`: expected missing fields, confirmation, or MCP elicitation behavior, if any.
- `expected_fee`: authoritative expected values copied from `data/fees_seed.demo.csv`.
- `expected_source`, `expected_pages`, `reference_contexts`: evidence labels taken from the named local PDFs. Page numbers are one-based PDF page numbers.
- `reference`: concise target answer or required facts. For finance records, monetary values are exact. For academic answers, it describes the claims and caveats that a response must preserve.

## Important notes

- These are expected outcomes, not measured model scores. Some cases intentionally describe desired behavior that may expose a defect in the current implementation.
- Fee cases use demo data only. Do not substitute real customer records into this committed file.
- Book titles and caveats reflect the contents of the local generated PDF. The PDF itself says to confirm current NCERT editions and school lists; the evaluation labels do not independently verify the live catalogue.
- The academic PDF references and excerpts were checked against the local PDFs. The book recommendation document uses page ranges because one Standard-and-subject record spans multiple pages.
- `reference_contexts` provide gold evidence for later RAG evaluation. The runner saves actual retrieved passages from the specialist evidence; RAG metrics must compare those passages against the gold contexts, not compare the expected excerpts to themselves.
- `expected_action_satisfied` checks the observable workflow behavior. It does not automatically judge the meaning, completeness, or groundedness of generated prose; review the saved answer and evidence, then add Ragas metrics in a later step.
- ROC AUC is not represented here because the supervisor currently returns a categorical route without per-intent confidence scores.

## Validation

Validate JSONL syntax with:

```bash
python -c 'import json; from pathlib import Path; p=Path("evals/datasets/support_eval.jsonl"); rows=[json.loads(line) for line in p.read_text().splitlines() if line.strip()]; print(f"{len(rows)} labeled cases")'
```
