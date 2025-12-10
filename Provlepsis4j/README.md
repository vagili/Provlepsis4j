# Neo4j Web App — Complete Stack (Frontend + Python Backend)

This zip contains everything you need to run a **browser app** that:
- Loads/creates a graph
- Projects to **GDS**
- Runs **embeddings** (FastRP, Node2Vec, GraphSAGE, HashGNN)
- Trains **Link Prediction** (Logistic Regression and/or Random Forest)
- Writes predicted edges `:REL_PRED(score)`
- Runs queries and visualizes predictions

## Option A — Run with Docker (includes local Neo4j with GDS)
```bash
docker compose up --build
# Frontend: http://localhost:5173
# Backend:  http://localhost:8080/health
# Neo4j UI: http://localhost:7474
```
In the UI:
1) **Data → Load Sample Dataset**
2) **Split → Create Split → Project to GDS**
3) **Embeddings → FastRP**
4) **Train & Predict → Train LP → Predict**
5) **Graph View → enter center node ID (e.g., 0) → Load Top-k**
6) **Metrics → Compute**

## Option B — Use AuraDS (managed Neo4j + GDS)
1. Edit `backend-py/.env` (copy from `.env.example`):
   ```
   NEO4J_URI=neo4j+s://<your-aura-host>.databases.neo4j.io
   NEO4J_USER=<user>
   NEO4J_PASSWORD=<password>
   NEO4J_DATABASE=neo4j
   FRONTEND_ORIGIN=http://localhost:5173
   ```
2. Start backend locally:
   ```bash
   cd backend-py
   pip install -r requirements.txt
   uvicorn app.main:app --reload --port 8080
   ```
3. Start frontend:
   ```bash
   cd frontend
   cp .env.example .env   # VITE_API=http://localhost:8080
   npm install
   npm run dev
   ```

## Endpoints
- `POST /ingest/sample` — load a tiny demo dataset
- `POST /split/create` — create `REL_TRAIN` / `REL_TEST`
- `POST /gds/graph/project` — project train graph
- `POST /emb/fastrp|node2vec|graphsage|hashgnn` — write embeddings
- `POST /lp/train` → add features + models (LR/RF) and **train**
- `POST /lp/predict` → write `REL_PRED(score)`
- `POST /metrics/lp` → precision/recall/F1 @ threshold
- `POST /metrics/topPredicted` → helper listing top-k predicted neighbors
- `POST /query/run` → ad-hoc Cypher (protect in prod)

Enjoy!
