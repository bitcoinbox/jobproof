# Project — LLM Evaluation Harness

> Fictional sample project for the persona "Robin Vega."

## What it is

A small evaluation harness that measures the quality of the support assistant and the
ticket parser, run in CI so prompt or model changes can't silently regress.

## How it works

- **Labeled sets:** curated `(input, expected)` pairs — for retrieval, the documents that
  *should* be retrieved; for extraction, the correct structured fields.
- **Metrics:** keyword/field match rate (precision + recall against the labeled answer),
  plus the assistant's own honest fit/confidence score tracked over time.
- **A/B arms:** every change is measured with vs. without the change (e.g. RAG vs. no-RAG,
  prompt v1 vs. v2) so improvements are quantified, not vibes.
- **CI gate:** the harness runs on every PR; a drop below threshold fails the build.

## My role

Built the metric functions (pure, unit-tested), the dataset format, and the CI wiring.

## Impact / lessons

- "I evaluate my LLM system" turned out to be the thing that separated our work from
  prompt-and-pray. Catching a 2-prompt regression before release paid for the harness many
  times over.
- Learned to keep metrics pure and deterministic so they're testable without a model call.

## Keywords

evaluation, evals, metrics, precision, recall, A/B testing, CI/CD, regression testing,
LLM quality, Python.
