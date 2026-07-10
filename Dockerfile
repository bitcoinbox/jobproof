FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Bake a populated demo into the image so the deployed dashboard is non-empty out of the
# box. This uses ONLY committed, fictional sample data (the Robin Vega persona + sample
# fixtures) — no personal data, no API key, no network. See .dockerignore: real resume,
# keys, DB, and corpus are never copied into the image.
RUN python -m src.cli demo --db jobsearch.db || true

# The web server is the default command. `serve` reads $PORT (Railway/Render set it),
# falling back to 8000. The CLI still works by overriding the command:
#   CLI:    docker run -e ANTHROPIC_API_KEY=$KEY jobproof run --source fixture --rag
#   Web UI: docker run -e ANTHROPIC_API_KEY=$KEY -p 8000:8000 jobproof   (default)
#   Self-hosted model: set AUTO_APPLY_LLM_BACKEND=local + AUTO_APPLY_LLM_BASE_URL/MODEL
ENTRYPOINT ["python", "-m", "src.cli"]
CMD ["serve", "--host", "0.0.0.0"]
