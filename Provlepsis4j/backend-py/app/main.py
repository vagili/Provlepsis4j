import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from .routers import databases

from .routers import gds, emb_fastrp, emb_node2vec, emb_graphsage, emb_hashgnn, lp, split, metrics, query, ingest, gds_meta, gds_context, gds_store, lp_iter
from .routers import config as config_router
from .routers import gds_project_from_db
from .routers import load_graph
from .routers import query

load_dotenv()

app = FastAPI(title="Neo4j ML API (Python)")

# CORS
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN, "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(config_router.router, prefix="/config", tags=["config"])
app.include_router(ingest.router, prefix="/ingest", tags=["ingest"])
app.include_router(gds.router, prefix="/gds", tags=["gds"])
app.include_router(emb_fastrp.router, prefix="/emb/fastrp", tags=["embeddings"])
app.include_router(emb_node2vec.router, prefix="/emb/node2vec", tags=["embeddings"])
app.include_router(emb_graphsage.router, prefix="/emb/graphsage", tags=["embeddings"])
app.include_router(emb_hashgnn.router, prefix="/emb/hashgnn", tags=["embeddings"])
app.include_router(lp.router, prefix="/lp", tags=["link-prediction"])
app.include_router(split.router, prefix="/split", tags=["split"])
app.include_router(metrics.router, prefix="/metrics", tags=["metrics"])
app.include_router(query.router, prefix="/query", tags=["query"])

app.include_router(gds_meta.router, prefix="/gds", tags=["gds-meta"])
app.include_router(gds_context.router, prefix="/gds", tags=["gds-context"])

app.include_router(gds_project_from_db.router, tags=["gds"])
app.include_router(gds_store.router, prefix="/gds", tags=["gds"])
app.include_router(load_graph.router, tags=["load"])
app.include_router(databases.router, tags=["db"])
app.include_router(lp_iter.router, prefix="/lp-iter", tags=["lp-iter"])

@app.get("/health")
def health():
    return {"status": "ok"}
