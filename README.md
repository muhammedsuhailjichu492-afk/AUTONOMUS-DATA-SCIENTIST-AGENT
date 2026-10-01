# Autonomous Data Scientist Agent

A ChatGPT-style web app for data science: upload a CSV, Excel, JSON, JSONL,
Parquet, or Feather dataset,
describe what you want to predict or understand, and an autonomous agent
handles profiling, cleaning, EDA, feature engineering, model selection,
training, evaluation, and reporting — end to end, explained in plain
language, with follow-up Q&A.

```
Data Profiling → Cleaning → EDA → Feature Engineering →
Model Selection → Training → Evaluation → Report
```

An LLM (Claude, via the Anthropic API) sits on top of that pipeline as an
agentic layer: it reviews the cleaning strategy and model shortlist, writes
the narrative summary, and powers follow-up chat. Every LLM step is
fail-soft — with no API key configured, the pipeline runs on rule-based
logic alone.

## Architecture

```
frontend/          Next.js + TypeScript + Tailwind — ChatGPT-style UI
backend/            FastAPI — REST + SSE streaming API, SQLAlchemy models
agent_core/         The data-science pipeline (unchanged — shared by both
                    the new backend and the legacy Streamlit app)
legacy_streamlit/   The original Streamlit prototype, kept runnable
sample_data/        Example dataset for trying the pipeline
```

`agent_core/` is the single source of truth for all data-science logic —
profiling, cleaning, EDA, feature engineering, model selection, training,
evaluation, reporting, and the LLM client/advisor/chat layer. Neither the
new backend nor the legacy Streamlit app duplicate any of it; both import
it directly.

### How a chat request becomes an analysis

```
User message (+ optional dataset upload)
        |
        v
POST /api/chat  (backend/app/api/chat.py)
        |
        v
run_agent_streaming()  (backend/app/agent/runner.py)
        |  wraps AutonomousDataScientistAgent.run() in a thread,
        |  emits progress/chart/model events over SSE as each
        |  pipeline stage (agent_core/orchestrator.py) completes
        v
Server-Sent Events -> frontend/lib/api.ts (SSE parser)
        |
        v
frontend/hooks/useChat.ts renders progress, dataset/model cards,
charts, and streamed narrative text into the chat thread
```

## Running it

### Backend (FastAPI)

```bash
cd backend
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # then fill in ANTHROPIC_API_KEY, SECRET_KEY, etc.
uvicorn app.main:app --reload --port 8000
```

SQLite is used by default (`backend/data/agent.db`); set `DATABASE_URL` in
`.env` to a `postgresql://...` URL for production — the SQLAlchemy layer
(`backend/app/database/`) works unmodified against either.

### Frontend (Next.js)

```bash
cd frontend
npm install
cp .env.local.example .env.local
npm run dev
```

Open `http://localhost:3000`. In dev, `next.config.mjs` proxies `/api/*` to
the FastAPI backend (`BACKEND_URL`, default `http://localhost:8000`), so no
CORS configuration is needed locally. In production, either keep that proxy
at your edge/CDN, or set `NEXT_PUBLIC_API_URL` so the frontend calls the
backend directly (and set `CORS_ORIGINS` in the backend `.env` accordingly).

### Legacy Streamlit app (kept for reference)

```bash
python -m streamlit run legacy_streamlit/app.py
```

Uses the same `agent_core/` package, run from the project root.

### CLI

```bash
python main.py --csv sample_data/customer_churn.csv --target Churn \
    --objective "Predict customer churn to prioritize retention outreach."
```

## Environment variables (backend)

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | `sqlite:///./data/agent.db` (dev) or `postgresql://...` (prod) |
| `ANTHROPIC_API_KEY` | Enables LLM-assisted advice, narration, and chat |
| `LLM_MODEL` | Defaults to `claude-sonnet-4-6` |
| `SECRET_KEY` | App secret — generate a real one for production |
| `CORS_ORIGINS` | Comma-separated list of allowed frontend origins |
| `UPLOAD_DIR` / `REPORTS_DIR` | Where datasets and generated reports are stored |
| `MAX_UPLOAD_SIZE_MB` | Upload size limit (default 50MB) |

See `backend/.env.example` for the full list.

## API

| Endpoint | Purpose |
|---|---|
| `POST /api/upload` | Upload a CSV, Excel, JSON, JSONL, Parquet, Feather, or MP4 file; datasets return metadata + preview, MP4 files are stored as media assets |
| `POST /api/chat` | SSE stream: runs the full pipeline or a follow-up chat turn |
| `GET /api/conversations?session_id=` | List conversations |
| `POST /api/conversations` | Create a conversation |
| `GET /api/conversations/{id}` | Conversation + message history |
| `DELETE /api/conversations/{id}` | Delete a conversation |
| `GET /api/datasets/{id}` | Dataset metadata |
| `GET /api/report/{conversation_id}/download` | Download `report.md` |
| `GET /api/media/{conversation_id}/download` | Download the latest uploaded MP4 |
| `GET /api/report/{conversation_id}/text` | Report as plain text |
| `GET /api/report/{conversation_id}/status` | Whether a run/report exists |

## Security notes

- The LLM never executes code or shell commands — it selects from a fixed
  set of Python tools in `agent_core/`, which perform all computation.
- Uploaded files are validated by extension and size, stored under a
  per-conversation UUID directory, and never trusted as executable input.
- No API keys are ever sent to or embedded in frontend code — the Anthropic
  key lives only in the backend's environment.

## Data science pipeline (`agent_core/`)

| Module | Stage |
|---|---|
| `profiler.py` | Profiles rows/columns/types/missing data, infers classification vs. regression |
| `cleaner.py` | Drops junk columns, imputes missing values, caps outliers |
| `eda.py` | Correlation heatmap, distributions, boxplots, target/categorical plots |
| `feature_engineer.py` | Builds the preprocessing pipeline, train/test split |
| `model_selector.py` | Shortlists candidate models by task type, size, and feature mix |
| `trainer.py` | Cross-validates every candidate, ranks a leaderboard |
| `evaluator.py` | Test-set metrics, confusion matrix / residuals plot |
| `reporter.py` | Assembles the full Markdown report |
| `llm_client.py` | Fail-soft Anthropic API wrapper (shared by every LLM feature) |
| `llm_advisor.py` | LLM review of cleaning strategy and model shortlist |
| `narrator.py` | LLM-written narrative summary |
| `chat.py` | `DataChatAgent` — grounded follow-up Q&A about a completed run |
| `orchestrator.py` | `AutonomousDataScientistAgent` — chains every stage end to end |
