# Evalio: Docker & Database Guide for Engineering

This guide explains the "What, Why, and How" of the Docker infrastructure and database architecture used in Evalio.

## 1. Why Docker?

Evalio is a complex AI Commerce Engine with multiple moving parts. We use Docker to:
- **Ensures Environment Parity**: "It works on my machine" is solved. Windows, Mac, and Linux developers all run the exact same environment.
- **pgvector Integration**: We need a specialized PostgreSQL version with `pgvector` for our semantic search. Docker makes this a one-line setup.
- **Service Orchestration**: Coordinating FastAPI, Redis (for sessions), and Celery (for scraping) requires a single source of truth (`docker-compose.yml`).

## 2. The Tech Stack (What?)

| Service | Technology | Port (Host) | Description |
| :--- | :--- | :--- | :--- |
| **Postgres** | `pgvector/pgvector:pg16` | **5433** | Vector database for products and embeddings. |
| **Redis** | `redis:7-alpine` | **6379** | Session cache and Celery message broker. |
| **Backend** | `FastAPI` | **8000** | The core LangGraph engine and API. |
| **Worker** | `Celery` | N/A | Background task processing (Scraping). |

> [!IMPORTANT]
> **Port Conflict Notice**: Most Windows machines have a local `postgres.exe` service running on port 5432. To avoid conflicts, we have remapped the Docker PostgreSQL to **5433**.

## 3. Getting Started (How?)

### Step 1: Start Infrastructure
```powershell
cd infra
docker-compose up -d
```

### Step 2: Initialize Database
The database must be initialized with the schema before the app can run.
```powershell
docker exec evalio-backend alembic upgrade head
```

### Step 3: Run the CLI
Use the CLI to test the integrated LangGraph nodes.
```powershell
python backend/app/cli.py
```

## 4. Use Cases in Evalio

1. **Semantic Search**: The `embeddings` table uses `pgvector` to find products based on cosine similarity to the user's query.
2. **Session Persistence**: Redis stores the `SessionData` object (active comparisons, last results) so the AI can remember history across turns.
3. **Data Freshness**: When the `DecisionMatrix` detects stale data, it triggers a Celery worker inside Docker to scrape updated prices.

## 5. Deployment Workflow

When the app is deployed (e.g., to AWS or DigitalOcean):
1. **Container Registry**: We build images for `backend` and `scraper`.
2. **Environment Variables**: The `.env` file is replaced by production secrets (Managed DB URL, Production OpenAI keys).
3. **Volumes**: `pg_data` and `redis_data` volumes are mounted to persistent storage to prevent data loss on restarts.
4. **Networking**: Containers communicate via internal Docker network names (e.g., `postgres:5432`) instead of `localhost`.

---

## 6. Collaboration & Sharing (For Teammates)

Yes, your teammate can and should use Docker. To work together seamlessly, you must share the following items that are **not** in Git:

### What to share with your teammate:
1. **The `.env` File**: This contains the `OPENAI_API_KEY` and database credentials. Do not commit this to Git! Share it via a secure channel (Slack, Keychain, etc.).
2. **Database Seed (Optional)**: If you have local data you want them to have, share a SQL dump:
   ```powershell
   docker exec ag-postgres pg_dump -U postgres evalio > evalio_dump.sql
   ```

### Teammate Setup Steps:
1. **Clone the Repo**: `git clone <repo-url>`
2. **Paste `.env`**: Place the shared `.env` in the root directory.
3. **Internal Port Check**: Ensure they don't have another service on port `5433` or `6379`.
4. **Boot Up**: 
   ```powershell
   cd infra
   docker-compose up -d
   ```
5. **Migrate**: `docker exec evalio-backend alembic upgrade head`

## 🚀 Moving to the Next Sprint?

**YES.** Sprint 14 (CLI Integration & Graph Wiring) is complete. 
- [x] LangGraph stubs replaced with real nodes.
- [x] Database schema established and verified. (Remapped to **5433**).
- [x] CLI fully functional with session persistence.

**Recommendation**: Proceed to **Sprint 15 / Feature Implementation**.
