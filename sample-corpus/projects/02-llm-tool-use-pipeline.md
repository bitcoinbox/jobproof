# Project — LLM Tool-Use Ticket Parser

> Fictional sample project for the persona "Robin Vega."

## What it is

A pipeline that turns messy inbound support tickets (HTML email, pasted logs, free text)
into clean, typed records the rest of the system can act on — built with the model's
**tool use / function calling**.

## How it works

- Each raw ticket is sent to **the LLM** with a single tool (`record_ticket`) whose
  `input_schema` defines the target record (category, severity, product area, summary,
  customer-impacted boolean, reproduction steps).
- `tool_choice` forces the call, so the model returns exactly one validated structured
  object instead of prose — no brittle regex parsing of free text.
- The structured records flow into the queue and analytics, and high-severity ones page
  on-call automatically.

## My role

Defined the tool schema, wrote the extraction prompt, and built the batch runner that
processes tickets with dedup and retry. Added a deterministic, no-LLM fallback parser for
tests so CI never needs network or an API key.

## Impact / lessons

- Replaced a fragile keyword-rules classifier; structured-output accuracy jumped and the
  code got simpler.
- Learned that forcing a tool is the cleanest way to get typed data out of an LLM — and
  that you still validate the model's output against your schema before trusting it.

## Keywords

tool use, function calling, structured outputs, JSON schema, LLM API, extraction,
classification, Python, batch processing.
