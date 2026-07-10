# Deploying JobProof

JobProof is a stateful FastAPI service (SQLite + a local vector index + headless-Chrome PDF
rendering), so it belongs on a container host — **Railway** or **Render**, not a serverless
platform like Vercel/Netlify. The repo ships a `Dockerfile`, a `railway.json`, and a
`.dockerignore`, and the image **bakes a populated demo** (fictional sample data) so the
dashboard is non-empty the moment it boots — no API key required.

## Railway (recommended)

1. Push this repo to GitHub (the public `jobproof` repo already is).
2. On [railway.com](https://railway.com): **New Project → Deploy from GitHub repo →** pick the repo.
   Railway auto-detects the `Dockerfile` and `railway.json` (health check on `/healthz`).
3. It builds and deploys. Open the generated URL → the cockpit loads, populated with the demo.
4. **Optional — enable tailoring/kits:** Project → **Variables** → add `ANTHROPIC_API_KEY`.
   Without it the dashboard, queue, diligence, and link-checker all still work; only the
   LLM-backed tailoring/kit buttons return a clean "set a key" message.
5. **Optional — run on your own models instead:** set
   `AUTO_APPLY_LLM_BACKEND=local`, `AUTO_APPLY_LLM_BASE_URL`, `AUTO_APPLY_LLM_MODEL`.

CLI from the same image (one-off jobs):
`railway run python -m src.cli report --digest`

## Render (alternative)

New → **Web Service** → connect the repo → Runtime **Docker**. Health check path `/healthz`.
Add `ANTHROPIC_API_KEY` under Environment if you want tailoring. The Docker `CMD` already
serves on `$PORT`, so no start command is needed.

## Privacy / data safety

- Deploy from the **public** repo only — it runs entirely on a fictional persona (Robin Vega).
- `.dockerignore` guarantees real data (`.env`, `*.db`, `.chroma/`, `experience/`,
  `master-resume.md`) is **never** copied into an image, even from a local build.
- The demo data baked at build time is the committed, invented sample set — no personal data.
- Your `ANTHROPIC_API_KEY` lives only in the host's env vars; it is never committed.

## Notes

- The container filesystem is ephemeral. For a public demo that's ideal (it resets to the
  clean fictional demo on redeploy). For real personal use, run it locally or attach a
  persistent volume mounted at the `AUTO_APPLY_DB` / `AUTO_APPLY_INDEX` paths.
