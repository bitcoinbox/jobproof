# Project — Platform Backend (pre-LLM foundation)

> Fictional sample project for the persona "Robin Vega."

## What it is

The core backend at Tide Pool Systems — the APIs, data model, and background processing
the product ran on, before any LLM work. The reason I can ship and operate the AI features
above, not just prototype them.

## How it works

- **APIs:** FastAPI services with typed request/response models and REST endpoints.
- **Data:** PostgreSQL schema design, migrations, and query tuning.
- **Async work:** Redis-backed queues and background workers for jobs that shouldn't block
  a request (emails, exports, third-party syncs).
- **Delivery:** Dockerized services, CI/CD with GitHub Actions, deploys to Railway, with
  health checks and basic metrics/logging.

## My role

Owned several services end to end: design, implementation, tests, deploy, and on-call.

## Impact / lessons

- This is the "can you actually run it in production" half of applied AI — queues,
  migrations, containers, monitoring. LLM features are only as good as the system around
  them.

## Keywords

FastAPI, REST API, PostgreSQL, Redis, queues, background workers, Docker, CI/CD,
GitHub Actions, Railway, Linux, monitoring, Python, TypeScript.
