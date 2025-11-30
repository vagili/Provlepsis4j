# app/routers/iterate.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Tuple, Optional

from ..db import run, run_data, current_database

# Reuse your existing modules; we bind them to the target DB at runtime.
from . import split as split_mod
from . import lp as lp_mod

router = APIRouter()

EMB_FAMILIES = ("FastRP", "Node2Vec", "GraphSAGE", "HashGNN")


class IterateBody(BaseModel):
    embeddingFamily: str = Field(..., description="FastRP | Node2Vec | GraphSAGE | HashGNN")
    embeddingProperty: str
    predictK: int = Field(default=100, ge=1)
    candidateMultiplier: int = Field(default=20, ge=2, le=200)
    probThreshold: float = Field(default=0.80, ge=0.0, le=1.0)
    negativeRatio: float = Field(default=1.0, ge=0.1, le=10.0)


def _predicted_db_name(base: str, family: str) -> str:
    return f"{base}-predicted-{family}"


def _db_exists(dbname: str) -> bool:
    try:
        rows = run_data(
            "SHOW DATABASE $db YIELD name, currentStatus RETURN currentStatus",
            {"db": dbname},
            db="system",
        )
        return bool(rows)
    except Exception:
        return False


def _with_db_bound(module, target_db: str):
    """
    Temporarily rebind a module's run/run_data/current_database to a specific DB.
    """
    class _Ctx:
        def __enter__(self_inner):
            self_inner._orig_run = module.run
            self_inner._orig_run_data = module.run_data
            self_inner._orig_cur = getattr(module, "current_database", None)

            def _r(q, p=None, **kw):
                return run(q, p or {}, db=target_db)

            def _rd(q, p=None, **kw):
                return run_data(q, p or {}, db=target_db)

            module.run = _r
            module.run_data = _rd
            if self_inner._orig_cur is not None:
                module.current_database = lambda: target_db
            return self_inner

        def __exit__(self_inner, exc_type, exc, tb):
            module.run = self_inner._orig_run
            module.run_data = self_inner._orig_run_data
            if self_inner._orig_cur is not None:
                module.current_database = self_inner._orig_cur
    return _Ctx()


def _next_counters(target_db: str) -> tuple[int, int]:
    """
    Read the current max timestamp/predicted in the *existing* predicted DB.
    """
    rows = run_data(
        """
        MATCH ()-[r]-()
        RETURN coalesce(max(r.timestamp), 0) AS max_ts,
               coalesce(max(r.predicted), 0) AS max_pred
        """,
        db=target_db,
    ) if _db_exists(target_db) else [{"max_ts": 0, "max_pred": 0}]
    max_ts = int(rows[0]["max_ts"] or 0)
    max_pred = int(rows[0]["max_pred"] or 0)
    return max_ts + 1, max_pred + 1


def _materialize_prediction_into_db(
    gname: str,
    target_db: str,
) -> Dict[str, Any]:
    """
    Persist the projected graph `gname` into `target_db` *without dropping* the DB:
      • Nodes are upserted by (:Entity {eid}) and enriched with props/labels.
      • Original edges (non-:PREDICTED) are ensured to exist; properties are left as-is.
      • New predicted edges are MERGEd and get `predicted=nextPred`, `timestamp=nextTs` **ON CREATE ONLY**.
      • Existing predicted edges keep their previous predicted/timestamp.
      • For PREDICTED we store **both directions** to match projection counts.
    """
    if not _db_exists(target_db):
        raise HTTPException(status_code=404, detail=f"Predicted DB '{target_db}' not found.")

    next_ts, next_pred = _next_counters(target_db)

    # Ensure node key constraint
    run(
        "CREATE CONSTRAINT node_eid_uniq IF NOT EXISTS FOR (n:Entity) REQUIRE n.eid IS UNIQUE",
        db=target_db,
    )

    # ---- Nodes from the current projection endpoints ----
    node_rows = run_data(
        """
        CALL gds.graph.relationships.stream($g)
        YIELD sourceNodeId, targetNodeId
        WITH gds.util.asNode(sourceNodeId) AS sN, gds.util.asNode(targetNodeId) AS tN
        WITH collect(sN) + collect(tN) AS nodes
        UNWIND nodes AS n
        WITH DISTINCT n
        RETURN elementId(n) AS eid, labels(n) AS labels, properties(n) AS props
        """,
        {"g": gname},
    )
    if node_rows:
        run(
            """
            UNWIND $rows AS row
            CALL apoc.merge.node(row.labels, {eid: row.eid}, row.props, row.props) YIELD node
            RETURN count(*) AS upserted
            """,
            {"rows": node_rows},
            db=target_db,
        )

    # ---- Relationship skeleton (distinct undirected pair per type) from projection ----
    rel_rows = run_data(
        """
        CALL gds.graph.relationships.stream($g)
        YIELD sourceNodeId, targetNodeId, relationshipType
        WITH gds.util.asNode(sourceNodeId) AS sN,
             gds.util.asNode(targetNodeId) AS tN,
             relationshipType AS type
        WITH elementId(sN) AS a, elementId(tN) AS b, type
        WITH CASE WHEN a < b THEN a ELSE b END AS sid,
             CASE WHEN a < b THEN b ELSE a END AS tid,
             type
        RETURN DISTINCT sid, tid, type
        """,
        {"g": gname},
    ) or []

    # 1) Ensure ORIGINAL edges exist (no prop clobbering)
    if rel_rows:
        run(
            """
            UNWIND $rows AS row
            WITH row WHERE row.type <> 'PREDICTED'
            MATCH (s:Entity {eid: row.sid})
            MATCH (t:Entity {eid: row.tid})
            CALL apoc.merge.relationship(s, row.type, {}, {}, t) YIELD rel
            RETURN count(*) AS ensured
            """,
            {"rows": rel_rows},
            db=target_db,
        )

    # 2) Add ONLY *new* predicted edges; keep old predicted edges untouched
    #    WRITE **BOTH** directions so the stored DB matches the projection counts.
    if rel_rows:
        run(
            """
            UNWIND $rows AS row
            WITH row WHERE row.type = 'PREDICTED'
            MATCH (s:Entity {eid: row.sid})
            MATCH (t:Entity {eid: row.tid})

            // FORWARD (s)->(t)
            // apoc.merge.relationship(start, type, relProps, onCreateProps, end)
            CALL apoc.merge.relationship(
              s, row.type, {}, {predicted: $nextPred, timestamp: $nextTs}, t
            ) YIELD rel AS r1
            WITH row, s, t, r1

            // REVERSE (t)->(s)
            CALL apoc.merge.relationship(
              t, row.type, {}, {predicted: $nextPred, timestamp: $nextTs}, s
            ) YIELD rel AS r2
            RETURN count(*) AS upserted
            """,
            {"rows": rel_rows, "nextTs": next_ts, "nextPred": next_pred},
            db=target_db,
        )

    # Summaries for the caller
    predicted_pairs = sum(1 for r in rel_rows if r.get("type") == "PREDICTED")
    orig_pairs = len(rel_rows) - predicted_pairs

    return {
        "target": target_db,
        "newTimestamp": next_ts,
        "newPredictedLevel": next_pred,
        "projectedPairsTotal": len(rel_rows),
        "projectedPairsOriginal": orig_pairs,
        "projectedPairsPredicted": predicted_pairs,
    }



@router.post("/run")
def iterate_run(body: IterateBody):
    # Resolve the predicted DB we should extend
    base = current_database()
    fam = body.embeddingFamily.strip()
    if fam not in EMB_FAMILIES:
        raise HTTPException(status_code=400, detail=f"Unknown embeddingFamily '{fam}'. Choose one of {EMB_FAMILIES}.")
    target_db = _predicted_db_name(base, fam)
    if not _db_exists(target_db):
        raise HTTPException(status_code=404, detail=f"Predicted DB '{target_db}' not found. Create the 1st-degree prediction first.")

    # 1) Split IN the target predicted DB (reuse your splitter; exclude strings for GDS safety)
    with _with_db_bound(split_mod, target_db):
        _orig_allowed = getattr(split_mod, "_ALLOWED_SCALARS", (int, float, bool, str))
        split_mod._ALLOWED_SCALARS = (int, float, bool)  # keep strings out of GDS
        try:
            split_mod.clear_feature_cache(db=target_db)
            split_mod.execute_master_split(
                split_mod.ExecuteSplitBody(
                    trainGraphName="trainGraph",
                    testGraphName="testGraph",
                    valGraphName=None,
                    testHoldout=0.10,
                    valHoldout=0.0,
                    ensureConnected=True,
                    includeEmbeddingProps=False,
                    reFreshFeatureCache=True,
                )
            )
        finally:
            split_mod._ALLOWED_SCALARS = _orig_allowed

    # 2) Train + predict IN the same DB (reuse lp.py), but do NOT let it create a new DB
    with _with_db_bound(lp_mod, target_db):
        # Monkey-patch materializer during this call to no-op (we'll persist ourselves)
        orig_mat = lp_mod._materialize_prediction_db  # type: ignore[attr-defined]
        lp_mod._materialize_prediction_db = lambda *a, **k: target_db  # type: ignore
        try:
            resp = lp_mod.lp_run(lp_mod.LPBody(
                embeddingProperty=body.embeddingProperty,
                trainGraphName="trainGraph",
                valGraphName=None,
                testGraphName="testGraph",
                negativeRatio=body.negativeRatio,
                predictK=body.predictK,
                candidateMultiplier=body.candidateMultiplier,
                probThreshold=body.probThreshold,
                outputGraphName="predictedGraph",
            ))
        finally:
            lp_mod._materialize_prediction_db = orig_mat  # type: ignore

    kept = int((resp.get("predicted") or {}).get("keptAboveThreshold", 0))

    # 3) Persist into the *existing* predicted DB without dropping; only add new PREDICTED edges
    info = _materialize_prediction_into_db("predictedGraph", target_db)

    return {
        "db": target_db,
        "keptAboveThreshold": kept,
        "added": kept,  # number of new predicted pairs this round
        "newTimestamp": info["newTimestamp"],
        "newPredictedLevel": info["newPredictedLevel"],
    }
