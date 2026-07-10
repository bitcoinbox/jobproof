# Evaluation set

A tiny, **fictional** labeled set used to measure whether RAG actually helps the
tailorer, run by `python -m src.cli eval`.

## Format (`labeled.jsonl`)

One JSON object per line:

| field | meaning |
|---|---|
| `id` | short label for the row |
| `job_file` | path (relative to the repo root) to the job posting text |
| `ideal_keywords` | keywords a strong, qualified candidate's tailored resume *should* surface |
| `min_expected_fit` | a sanity floor for the fit score (0 for the deliberate non-fit row) |

## Metric

For each job we tailor the resume twice — once with the full master profile, once with
RAG-retrieved experience — and compute:

- **keyword-match rate** = `|ideal ∩ surfaced| / |ideal|` (recall of the ideal keyword
  set, case-insensitive, over the tailored resume + matched keywords)
- **fit_score** = the model's own honest 0–100 self-assessment

We aggregate the mean per arm and report the **delta** (RAG − full-profile).

## Honest caveats

This set is small and fictional: it demonstrates the harness and produces a reproducible
number on the sample corpus — it is **not** a benchmark. On a one-page corpus, RAG often
ties the full-profile path (retrieval pays off as the corpus grows). The point is that the
system is *measured*, and the harness tells you which regime you're in. The
`senior-ml-research-phd` row is included on purpose to confirm the scorer reports a low
fit for a role the persona genuinely doesn't match.
