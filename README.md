# SIRA — FloatChat: Ocean Intelligence Platform

Transform raw Argo float observations into searchable, explainable ocean insights. SIRA combines scalable data processing, retrieval‑augmented generation (RAG), and an API surface that supports both curated facts and user‑provided datasets.

## Overview
- FastAPI backend exposing chat and data endpoints on `:8000`.
- Data pipeline: NetCDF → aggregation → anomaly labeling → semantic summaries → embeddings.
- RAG over semantic summaries using `Chroma` + OpenAI embeddings + Gemini chat.
- SQL QA over a SQLite database generated from semantic CSV via LLM SQL generation.
- Per‑user session and storage for uploads and personalized retrieval.
- Optional Streamlit UI pages for landing and graphs.

## Key Features
- Multi‑source chat:
  - `facts` from curated buoy text (`Buoys_data.txt`).
  - `netcdf` via RAG over processed Argo profiles.
  - `db` via LLM‑generated SQL executed against SQLite.
- Per‑user workflow:
  - Session‑scoped uploads of NetCDF (`/user/upload`) and CSV datasets (`/user/upload/dataset`).
  - User‑specific embeddings and DB built on upload.
  - Download processed CSV (`/user/download`).
- Visualization data APIs for region profiles, ready for frontend plotting.
- Dockerized deployment targeting Python 3.11.

## Architecture
- Backend: `FastAPI` (`main.py`)
- Pipeline (`integration.py`):
  - `concatfiles.main_process`: aggregate NetCDF into binned CSV (`pressure_bin`, `temp_adjusted`, `psal_adjusted`, metadata).
  - `analyze.data_transformation`: interpolate, region labeling, stats, z‑scores, anomaly flags; export final CSV.
  - `sematics.run_semantics`: generate human‑readable sentences per row; export semantic CSV.
  - `chunks.save_embeddings`: embed semantic sentences into `Chroma` (global and per‑user collections).
  - `dbagent.initialize_db`: load semantic CSV into SQLite (`argo_profiles` table).
- Retrieval:
  - `chunks.answer_query`: similarity search + Gemini answer.
  - `dbagent.query_db_q`: prompt LLM (OpenAI) to synthesize SQL; run on SQLite; return table.
- Curated facts:
  - `text_embeddings`: embed `Buoys_data.txt` and answer questions via Gemini.
- Optional UI:
  - `pages/landing.py`, `pages/graphs.py` (Streamlit) for marketing and basic plots.

## Environment Variables
- `OPENAI_API_KEY` — used for embeddings and LLM SQL generation.
- `GOOGLE_API_KEY` — used by Gemini via LangChain.
- `SESSION_SECRET_KEY` — session cookie signing (optional; random if not set).

Create a `.env` file:
```
OPENAI_API_KEY=your_openai_key
GOOGLE_API_KEY=your_google_key
SESSION_SECRET_KEY=replace_with_random_hex
```

## Getting Started
1) Python setup
- Install Python 3.11+
- Create a virtual environment and install dependencies:
```
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```
- Run the API:
```
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

2) Docker
- Build and run:
```
docker build -t sira .
docker run --rm -p 8000:8000 --env-file .env sira
```

## Data Workflow
- NetCDF ingestion:
  - Place `.nc` files under `uploads/<user_id>/argo_nc_files/` (automatically created on upload).
  - Or download programmatically using `automated_script.py` (reads `data_store.txt` and saves to `argo_nc_files/`).
- Processing steps (triggered by `/user/upload`):
  - Aggregate: `argo_profiles_binned_1.csv`
  - Enrich & label anomalies: `argo_profiles_final.csv`
  - Semantic summaries: `argo_semantic_summary_1.csv`
  - Embeddings: persisted in `./chroma_db`
  - SQLite DB: `uploads/<user_id>/argo.db` (table `argo_profiles`)

## API Reference
Base URL: `http://localhost:8000`

- Health
  - `GET /` → `{"message":"Multi-Model Chat API is running!"}`

- Sessions
  - `GET /user/session/new` → returns `{"user_id": "<uuid>"}` and sets session cookie.

- Uploads
  - `POST /user/upload` (form-data `files` list of `.nc`) → processes and indexes user data.
  - `POST /user/upload/dataset` (form-data `file` `.csv`) → store a CSV for model testing.

- Chat (global)
  - `POST /chat/facts` → curated buoy knowledge. Body: `{"question":"..."}`
  - `POST /chat/netcdf` → RAG over global semantic summaries. Body: `{"question":"..."}`
  - `POST /chat/db` → SQL QA over global SQLite. Body: `{"question":"..."}`

- Chat (per user)
  - `POST /user/chat/netcdf` → RAG filtered by `user_id`.
  - `POST /user/chat/db` → SQL QA against per‑user DB.

- Downloads
  - `GET /user/download` → download user’s `argo_semantic_summary_1.csv` if present.

- Visualization data
  - `GET /profiles/regions`
  - `GET /profiles/data?region=<region>`
  - `GET /users/profiles/regions`
  - `GET /users/profiles/data?region=<region>`

Example `curl`:
```
curl -s http://localhost:8000/user/session/new

curl -s -X POST "http://localhost:8000/chat/netcdf" \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"Where do we see high salinity at 500 dbar?\"}"
```

## Frontend Integration
- CORS allows:
  - `http://localhost:5173`
  - `https://sira-frontend-our-data.vercel.app`
  - `https://sira-frontend-user-data.vercel.app`
- Streamlit demo:
```
streamlit run landing.py
```
Graphs page expects `argo_profiles_final.csv` in repo root.

## Project Structure
- `main.py` — FastAPI app and endpoints.
- `integration.py` — orchestrates ingestion, transformation, semantics, embeddings, DB.
- `concatfiles.py` — NetCDF → binned CSV (Dask, Xarray).
- `analyze.py` — interpolation, region mapping, stats, z‑scores, anomaly flags.
- `sematics.py` — sentence generation for semantic summaries.
- `chunks.py` — embeddings and Gemini‑based RAG QA; persistent `Chroma` stores.
- `dbagent.py` — SQLite initialization; LLM‑generated SQL executor.
- `text_embeddings.py` — curated buoy facts embeddings + QA.
- `automated_script.py` — download Argo NetCDF from NCEI by index file.
- `pages/` — Streamlit landing and graphs.
- `Dockerfile`, `.dockerignore`, `.gitignore`, `requirements.txt`.

## Notes and Constraints
- Requires valid `OPENAI_API_KEY` and `GOOGLE_API_KEY`.
- Embedding creation for `Buoys_data.txt` runs on import in `text_embeddings.py`.
- `feature_detect.py` contains anomaly detection and event classification utilities; the `/user/test/model` endpoint currently returns a placeholder string.

## Running in Production
- Set strong `SESSION_SECRET_KEY`.
- Use persistent storage volumes for `uploads/` and `./chroma_db`.
- Configure HTTPS and correct `SameSite=None` cookies at the proxy layer.
