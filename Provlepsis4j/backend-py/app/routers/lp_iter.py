# app/routers/lp_iter.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import sys
import math
import datetime
import re
from .emb_graphsage import _resolve_graphsage_prefix
from ..db import run, run_data, current_database
from . import split as split_mod
from neo4j.exceptions import Neo4jError
# ----------------- sklearn import just for iterative LP -----------------

try:
    from sklearn.linear_model import LogisticRegression
except Exception:  # pragma: no cover
    LogisticRegression = None  # handled by _assert_sklearn_iter()

router = APIRouter()

EMB_FAMILIES = ("FastRP", "Node2Vec", "GraphSAGE", "HashGNN")

_PROP_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


# ----------------- Debug helper -----------------

def _dbg_iter(tag: str, **fields: Any) -> None:
    """
    Tiny debug helper just so we can see what's happening in docker logs.
    """
    try:
        db = current_database()
    except Exception:
        db = "<unknown>"

    ts = datetime.datetime.utcnow().isoformat()
    msg = " ".join(f"{k}={repr(v)}" for k, v in fields.items())
    line = f"{ts} [LP_ITER DEBUG] {tag} db={db} {msg}"
    print(line, file=sys.stderr, flush=True)


# ----------------- DB binding helper (for split only) -----------------

def _with_db_bound(module, target_db: str):
    """
    Temporarily rebind a module's run/run_data to point at `target_db`.
    Used for calling split.py *inside* the chosen predicted DB.
    """
    class _Ctx:
        def __enter__(self_inner):
            # Save originals
            self_inner._orig_run = getattr(module, "run", None)
            self_inner._orig_run_data = getattr(module, "run_data", None)

            # Wrappers that always send queries to `target_db`
            def _r(q, p=None, **kw):
                return run(q, p or {}, db=target_db)

            def _rd(q, p=None, **kw):
                return run_data(q, p or {}, db=target_db)

            if self_inner._orig_run is not None:
                module.run = _r
            if self_inner._orig_run_data is not None:
                module.run_data = _rd

            return self_inner

        def __exit__(self_inner, exc_type, exc, tb):
            # Restore originals
            if getattr(self_inner, "_orig_run", None) is not None:
                module.run = self_inner._orig_run
            if getattr(self_inner, "_orig_run_data", None) is not None:
                module.run_data = self_inner._orig_run_data

    return _Ctx()


# ----------------- Request model -----------------

class IterateBody(BaseModel):
    embeddingFamily: str = Field(..., description="FastRP | Node2Vec | GraphSAGE | HashGNN")
    embeddingProperty: str
    predictK: int = Field(default=100, ge=1)
    candidateMultiplier: int = Field(default=20, ge=2, le=200)
    probThreshold: float = Field(default=0.80, ge=0.0, le=1.0)
    negativeRatio: float = Field(default=1.0, ge=0.1, le=10.0)


# ----------------- Small utility helpers -----------------

def _predicted_db_name(base: str, family: str) -> str:
    """
    Given the current base DB and the chosen embedding family (FastRP, Node2Vec, ...),
    compute the predicted-db name we will iterate on.
    """
    name = f"{base}-predicted-{family}"
    _dbg_iter("_predicted_db_name", base=base, family=family, dbname=name)
    return name


def _db_exists(dbname: str) -> bool:
    """
    Just to sanity-check that the predicted DB actually exists.
    """
    try:
        rows = run_data(
            "SHOW DATABASE $db YIELD name, currentStatus RETURN currentStatus",
            {"db": dbname},
            db="system",
        )
        exists = bool(rows)
        _dbg_iter("_db_exists", dbname=dbname, exists=exists)
        return exists
    except Exception as e:
        _dbg_iter("_db_exists.error", dbname=dbname, error=str(e))
        return False


def _compute_level_and_ts(dbname: str) -> Tuple[int, int]:
    """
    Read current predicted level (max r.predicted, default 0)
    and max timestamp (default 0) for the chosen predicted DB.
    """
    _dbg_iter("_compute_level_and_ts.start", dbname=dbname)
    rows = run_data(
        """
        MATCH ()-[r]-()
        RETURN coalesce(max(r.timestamp),0) AS max_ts,
               coalesce(max(r.predicted),0) AS max_pred
        """,
        db=dbname,
    )
    if not rows:
        _dbg_iter("_compute_level_and_ts.empty", dbname=dbname)
        return (0, 0)
    max_ts = int(rows[0]["max_ts"] or 0)
    max_pred = int(rows[0]["max_pred"] or 0)
    _dbg_iter(
        "_compute_level_and_ts.done",
        dbname=dbname,
        max_ts=max_ts,
        max_pred=max_pred,
    )
    return max_pred, max_ts


def _assert_sklearn_iter() -> None:
    if LogisticRegression is None:
        _dbg_iter("_assert_sklearn_iter.missing")
        raise HTTPException(
            status_code=500,
            detail=(
                "scikit-learn is not installed in the backend image. "
                "Add it to requirements.txt (e.g., scikit-learn==1.4.*) and rebuild."
            ),
        )


def _sanitize_prop(name: str) -> str:
    if not _PROP_RE.match(name):
        _dbg_iter("_sanitize_prop.invalid", name=name)
        raise HTTPException(status_code=400, detail=f"Invalid embedding property name: {name}")
    return name


def _hadamard(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a * b


# ----------------- Embedding step on trainGraph -----------------

def _infer_embedding_dimension(dbname: str, prop: str, default: int = 128) -> int:
    """
    Try to infer the embedding dimension from an existing vector property in `dbname`.
    If none exists, fall back to `default`.
    """
    try:
        rows = run_data(
            f"""
            MATCH (n)
            WHERE n.{prop} IS NOT NULL AND size(n.{prop}) > 0
            RETURN size(n.{prop}) AS dim
            LIMIT 1
            """,
            db=dbname,
        )
        if rows and rows[0].get("dim") is not None:
            dim = int(rows[0]["dim"])
            _dbg_iter("_infer_embedding_dimension.found", dbname=dbname, prop=prop, dim=dim)
            return dim
    except Exception as e:
        _dbg_iter("_infer_embedding_dimension.error", dbname=dbname, prop=prop, error=str(e))
    _dbg_iter("_infer_embedding_dimension.default", dbname=dbname, prop=prop, dim=default)
    return default

# ---- HashGNN helpers for iterative LP ----

def _sanitize_hashgnn_prop(name: str) -> str:
    """
    Keep only alnum + '_' for safety. If it ends up empty, fall back to 'HashGNN'.
    """
    cleaned = "".join(ch for ch in str(name) if ch.isalnum() or ch == "_")
    return cleaned or "HashGNN"


def _resolve_hashgnn_stream_for_db(dbname: str) -> str:
    """
    Ensure HashGNN is available in this DB and return the stream procedure name.
    Currently we only use gds.hashgnn.stream.
    """
    rows = run_data(
        "CALL gds.list() YIELD name "
        "WHERE name IN ['gds.hashgnn.stream'] "
        "RETURN collect(name) AS present",
        {},
        db=dbname,
    )
    present = set(rows[0]["present"] if rows else [])
    if "gds.hashgnn.stream" in present:
        return "gds.hashgnn.stream"

    raise HTTPException(
        status_code=400,
        detail=(
            "HashGNN procedure 'gds.hashgnn.stream' is not available in this database. "
            "Check your GDS version/edition and `CALL gds.list()`."
        ),
    )


# ---- GraphSAGE helpers for iterative LP ----
def _resolve_graphsage_prefix_for_db(dbname: str) -> str:
    """
    Find whether gds.graphSage.* or gds.beta.graphSage.* is available in this DB
    and return the common prefix (e.g. 'gds.graphSage' or 'gds.beta.graphSage').
    """
    names = [
        "gds.graphSage.train",
        "gds.beta.graphSage.train",
        "gds.graphSage.write",
        "gds.beta.graphSage.write",
    ]
    rows = run_data(
        "CALL gds.list() YIELD name "
        "WHERE name IN $names "
        "RETURN collect(name) AS present",
        {"names": names},
        db=dbname,
    )
    present = set(rows[0]["present"] if rows else [])
    train = (
        "gds.graphSage.train"
        if "gds.graphSage.train" in present
        else ("gds.beta.graphSage.train" if "gds.beta.graphSage.train" in present else None)
    )
    write = (
        "gds.graphSage.write"
        if "gds.graphSage.write" in present
        else ("gds.beta.graphSage.write" if "gds.beta.graphSage.write" in present else None)
    )
    if not train or not write:
        raise HTTPException(
            status_code=400,
            detail=(
                "GraphSAGE procedures not found in this database. "
                "Your GDS version/edition may not include GraphSAGE "
                "(look for gds.graphSage.* or gds.beta.graphSage.* in CALL gds.list())."
            ),
        )
    # We return the common prefix: 'gds.graphSage' or 'gds.beta.graphSage'
    return train.rsplit(".", 1)[0]


def _graphs_age_model_list_proc_for_db(dbname: str) -> str:
    """
    Detect whether gds.model.list or gds.beta.model.list exists.
    """
    rows = run_data(
        "CALL gds.list() YIELD name "
        "WHERE name IN ['gds.model.list','gds.beta.model.list'] "
        "RETURN collect(name) AS present",
        {},
        db=dbname,
    )
    present = set(rows[0]["present"] if rows else [])
    if "gds.model.list" in present:
        return "gds.model.list"
    if "gds.beta.model.list" in present:
        return "gds.beta.model.list"
    # Very old GDS version – treat as "no model support"
    raise HTTPException(
        status_code=400,
        detail=(
            "GDS model catalog procedures (gds.model.list / gds.beta.model.list) "
            "are not available in this database; cannot manage GraphSAGE models."
        ),
    )


def _graphs_age_model_exists(dbname: str, model_name: str) -> bool:
    """
    Return True if a model with the given name exists in the GDS catalog for this DB.
    """
    proc = _graphs_age_model_list_proc_for_db(dbname)
    rows = run_data(
        f"CALL {proc}() YIELD modelName "
        "WHERE modelName = $name "
        "RETURN modelName",
        {"name": model_name},
        db=dbname,
    )
    return bool(rows)


def _graphs_age_discover_feature_props(dbname: str, graph: str) -> List[str]:
    """
    Same spirit as emb_graphsage._discover_feature_props, but scoped to a specific DB.
    We:
      - sample a node from the projection
      - take numeric / numeric-list-like properties
      - exclude embedding-like props
      - validate they exist in the projection
    """
    # Step 1: sample keys from a projected node
    rows = run_data(
        """
        WITH $g AS g
        CALL gds.degree.stream(g) YIELD nodeId
        WITH g, nodeId LIMIT 1
        WITH g, gds.util.asNode(nodeId) AS n
        RETURN [p IN keys(n)
                WHERE NOT p IN ['FastRP','Node2Vec','GraphSAGE','HashGNN']
                  AND NOT p STARTS WITH 'embedding_'
                  AND (
                    toFloatOrNull(n[p]) IS NOT NULL
                    OR (n[p] IS LIST AND size([x IN n[p] WHERE x IS NULL OR toFloatOrNull(x) IS NOT NULL]) = size(n[p]))
                  )
               ] AS numericProjectedProps
        """,
        {"g": graph},
        db=dbname,
    )
    props = rows[0]["numericProjectedProps"] if rows else []

    # Step 2: exclude obvious embedding props and internal keys (mirroring emb_graphsage)
    EMBEDDING_PREFIXES = ["fastrp_", "node2vec_", "graphsage_", "hashgnn_"]
    EMBEDDING_EXACTS = {"FastRP", "Node2Vec", "GraphSAGE", "HashGNN"}
    INTERNAL_KEYS = {"_id", "_tmp", "_ts", "id", "ID"}

    def _exclude_embedding_props(props_in: List[str]) -> List[str]:
        out: List[str] = []
        for p in props_in:
            if p in INTERNAL_KEYS or p in EMBEDDING_EXACTS:
                continue
            if any(p.startswith(pref) for pref in EMBEDDING_PREFIXES):
                continue
            out.append(p)
        return out

    props = _exclude_embedding_props(props)

    # Step 3: filter to “supported-looking” values (same spirit as emb_graphsage)
    _ALLOWED_SCALARS = (int, float, bool, str)

    def _looks_supported_value(v) -> bool:
        if v is None:
            return True
        if isinstance(v, _ALLOWED_SCALARS):
            return True
        if isinstance(v, list):
            for x in v:
                if x is None:
                    continue
                if not isinstance(x, (int, float)):
                    return False
            return True
        return False

    def _filter_supported_props(props_in: List[str]) -> List[str]:
        safe: List[str] = []
        for p in props_in:
            rows2 = run_data(
                "WITH $k AS k MATCH (n) WHERE n[k] IS NOT NULL RETURN n[k] AS v LIMIT 500",
                {"k": p},
                db=dbname,
            )
            if not rows2:
                safe.append(p)
                continue
            ok = True
            for r2 in rows2:
                if not _looks_supported_value(r2.get("v")):
                    ok = False
                    break
            if ok:
                safe.append(p)
        return safe

    props = _filter_supported_props(props)

    # Step 4: keep only those that actually exist in the projection
    rows3 = run_data(
        """
        WITH $g AS g, $props AS props
        UNWIND props AS p
        CALL {
          WITH g, p
          CALL gds.graph.nodeProperties.stream(g, [p])
          YIELD nodeId
          RETURN p AS ok
          LIMIT 1
        }
        RETURN collect(ok) AS okProps
        """,
        {"g": graph, "props": props},
        db=dbname,
    )
    return rows3[0]["okProps"] if rows3 else []


def _ensure_graphsage_model_for_db(
    dbname: str,
    graph: str,
    model_name: str,
    dim: int,
) -> str:
    """
    Ensure a GraphSAGE model with the given name exists in this DB.

    - If the model already exists (e.g. user ran initial GraphSAGE prediction),
      we just re-use it.
    - If not, we discover featureProperties and train a new model on `graph`.
    - Returns the GraphSAGE prefix (e.g. 'gds.graphSage' or 'gds.beta.graphSage').
    """
    prefix = _resolve_graphsage_prefix_for_db(dbname)

    if _graphs_age_model_exists(dbname, model_name):
        _dbg_iter(
            "_ensure_graphsage_model_for_db.exists",
            dbname=dbname,
            graph=graph,
            model=model_name,
        )
        return prefix

    # Need to train a new model in this DB
    used_features = _graphs_age_discover_feature_props(dbname, graph)
    if not used_features:
        raise HTTPException(
            status_code=400,
            detail=(
                "No suitable numeric feature properties were found in the in-memory projection "
                f"'{graph}' of database '{dbname}' to train a GraphSAGE model. "
                "Either supply feature properties explicitly in the initial GraphSAGE run, "
                "or ensure nodes have numeric / numeric-list properties."
            ),
        )

    train_cfg: Dict[str, Any] = {
        "modelName": model_name,
        "featureProperties": used_features,
        "aggregator": "mean",
        "sampleSizes": [25, 10],
        "embeddingDimension": int(dim),
        "epochs": 5,
        "learningRate": 0.01,
    }

    _dbg_iter(
        "_ensure_graphsage_model_for_db.train",
        dbname=dbname,
        graph=graph,
        model=model_name,
        cfg=train_cfg,
    )
    run_data(
        f"CALL {prefix}.train($g, $cfg)",
        {"g": graph, "cfg": train_cfg},
        db=dbname,
    )
    _dbg_iter(
        "_ensure_graphsage_model_for_db.trained",
        dbname=dbname,
        graph=graph,
        model=model_name,
    )
    return prefix

def _compute_embeddings_for_family(
    dbname: str,
    family: str,
    prop: str,
    graph_name: str = "trainGraph",
) -> Dict[str, Any]:
    """
    STEP 1 of the LP part:
      Recompute node embeddings on `graph_name` in `dbname`,
      using the family-specific GDS proc and writing to `prop`.
    """
    dim = _infer_embedding_dimension(dbname, prop, default=128)
    _dbg_iter(
        "_compute_embeddings_for_family.start",
        dbname=dbname,
        family=family,
        prop=prop,
        dim=dim,
        graph_name=graph_name,
    )

    # ---------- FastRP ----------
    if family == "FastRP":
        cfg = {
            "embeddingDimension": dim,
            "writeProperty": prop,
        }
        rows = run_data(
            """
            CALL gds.fastRP.write($g, $cfg)
            YIELD nodePropertiesWritten, nodeCount, computeMillis, writeMillis
            RETURN nodePropertiesWritten, nodeCount, computeMillis, writeMillis
            """,
            {"g": graph_name, "cfg": cfg},
            db=dbname,
        )
        info = rows[0] if rows else {}
        _dbg_iter("_compute_embeddings_for_family.fastrp_done", dbname=dbname, info=info)
        return {"family": family, "config": cfg, "info": info}

    # ---------- Node2Vec ----------
    if family == "Node2Vec":
        cfg = {
            "embeddingDimension": dim,
            "walkLength": 80,
            "walksPerNode": 10,
            "returnFactor": 1.0,
            "inOutFactor": 1.0,
            "writeProperty": prop,
        }
        rows = run_data(
            """
            CALL gds.node2vec.write($g, $cfg)
            YIELD nodePropertiesWritten, nodeCount, computeMillis, writeMillis
            RETURN nodePropertiesWritten, nodeCount, computeMillis, writeMillis
            """,
            {"g": graph_name, "cfg": cfg},
            db=dbname,
        )
        info = rows[0] if rows else {}
        _dbg_iter("_compute_embeddings_for_family.n2v_done", dbname=dbname, info=info)
        return {"family": family, "config": cfg, "info": info}

    # ---------- GraphSAGE ----------
    if family == "GraphSAGE":
        # We always use the same modelName as emb_graphsage.py by default
        model_name = "sage_model"

        # Ensure that a model with this name exists in THIS database.
        # If user already ran the initial GraphSAGE prediction in this run,
        # we just re-use it. If not, but the DB/projection exist, we train now.
        prefix = _ensure_graphsage_model_for_db(
            dbname=dbname,
            graph=graph_name,
            model_name=model_name,
            dim=dim,
        )

        write_cfg: Dict[str, Any] = {
            "modelName": model_name,
            "writeProperty": prop,
        }

        try:
            rows = run_data(
                f"""
                CALL {prefix}.write($g, $cfg)
                YIELD nodePropertiesWritten, computeMillis, writeMillis
                RETURN nodePropertiesWritten, computeMillis, writeMillis
                """,
                {"g": graph_name, "cfg": write_cfg},
                db=dbname,
            )
        except Neo4jError as e:
            msg = getattr(e, "message", str(e))
            # If, for some reason, model lookup missed but server says the model doesn't exist,
            # surface a clean 400 to the frontend instead of a 500.
            if "Model with name" in msg and "does not exist" in msg:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"GraphSAGE model '{model_name}' does not exist in database '{dbname}'. "
                        "Tried to auto-create it but the operation failed. "
                        "Please run the initial GraphSAGE /trainWrite step for this DB."
                    ),
                )
            raise

        info = rows[0] if rows else {}
        _dbg_iter(
            "_compute_embeddings_for_family.graphsage_done",
            dbname=dbname,
            model=model_name,
            writeProperty=prop,
            info=info,
        )
        return {"family": family, "config": write_cfg, "info": info}


    # ---------- HashGNN ----------
    if family == "HashGNN":
        """
        Iterative HashGNN embedding step.

        We *do not* use gds.hashgnn.write (which may not exist).
        Instead we always call gds.hashgnn.stream with generateFeatures,
        then SET the embedding onto the chosen property.
        """
        proc = _resolve_hashgnn_stream_for_db(dbname)
        write_prop = _sanitize_hashgnn_prop(prop)

        # Simple defaults – you can tune these if you like.
        it = 10
        dens = 10  # embeddingDensity / densityLevel
        params = {
            "g": graph_name,
            "it": it,
            "dim": int(dim),
            "dens": dens,
        }

        cypher = (
            f"CALL {proc}($g, {{"
            "  iterations: $it,"
            "  outputDimension: $dim,"
            "  embeddingDensity: $dens,"
            "  generateFeatures: { dimension: $dim, densityLevel: $dens }"
            "}) "
            "YIELD nodeId, embedding "
            "WITH gds.util.asNode(nodeId) AS n, [x IN embedding | toFloat(x)] AS v "
            f"SET n.{write_prop} = v "
            "RETURN count(*) AS wrote"
        )

        _dbg_iter(
            "_compute_embeddings_for_family.hashgnn_start",
            dbname=dbname,
            graph=graph_name,
            prop=write_prop,
            dim=dim,
            it=it,
            dens=dens,
            proc=proc,
        )

        rows = run_data(cypher, params, db=dbname)
        wrote = int(rows[0]["wrote"]) if rows else 0

        info = {
            "wrote": wrote,
            "iterations": it,
            "outputDimension": int(dim),
            "embeddingDensity": dens,
            "generateFeatures": {"dimension": int(dim), "densityLevel": dens},
            "writeProperty": write_prop,
        }

        _dbg_iter(
            "_compute_embeddings_for_family.hashgnn_done",
            dbname=dbname,
            graph=graph_name,
            prop=write_prop,
            info=info,
        )

        return {"family": family, "config": info, "info": info}

    # ---------- Fallback ----------
    _dbg_iter("_compute_embeddings_for_family.unsupported", dbname=dbname, family=family)
    raise HTTPException(
        status_code=400,
        detail=(
            "Iterative LP currently supports 'FastRP', 'Node2Vec', 'GraphSAGE' "
            f"and 'HashGNN' only. Got embeddingFamily='{family}'."
        ),
    )



# ----------------- Embeddings + pairs helpers inside chosen_db -----------------

def _load_embeddings_from_db(dbname: str, prop: str) -> Dict[str, np.ndarray]:
    """
    Read elementId -> embedding (numpy array) for all nodes in `dbname`
    where property exists.
    """
    _dbg_iter("_load_embeddings_from_db.start", dbname=dbname, prop=prop)
    rows = run_data(
        f"""
        MATCH (n)
        WHERE n.{prop} IS NOT NULL
        RETURN elementId(n) AS eid, n.{prop} AS emb
        """,
        db=dbname,
    )
    embs: Dict[str, np.ndarray] = {}
    for r in rows:
        v = r["emb"]
        if isinstance(v, list):
            embs[r["eid"]] = np.array(v, dtype=float)
    _dbg_iter("_load_embeddings_from_db.done", dbname=dbname, prop=prop, count=len(embs))
    return embs


def _get_train_positive_pairs(dbname: str, graph_name: str = "trainGraph") -> List[Tuple[str, str]]:
    """
    STEP 2: Get positive edge pairs from trainGraph in `dbname`.
    Returns UNDIRECTED canonical pairs (s < t) as elementId strings.
    """
    _dbg_iter("_get_train_positive_pairs.start", dbname=dbname, graph_name=graph_name)
    rows = run_data(
        """
        CALL gds.graph.relationships.stream($g)
        YIELD sourceNodeId, targetNodeId
        WITH gds.util.asNode(sourceNodeId) AS sN,
             gds.util.asNode(targetNodeId) AS tN
        WITH elementId(sN) AS a, elementId(tN) AS b
        WITH CASE WHEN a < b THEN a ELSE b END AS s,
             CASE WHEN a < b THEN b ELSE a END AS t
        RETURN DISTINCT s, t
        """,
        {"g": graph_name},
        db=dbname,
    )
    pairs = [(r["s"], r["t"]) for r in rows]
    _dbg_iter("_get_train_positive_pairs.done", dbname=dbname, graph_name=graph_name, count=len(pairs))
    return pairs


def _sample_negative_pairs(dbname: str, prop: str, k: int) -> List[Tuple[str, str]]:
    """
    STEP 4 (part): Sample ~k UNDIRECTED non-edges from `dbname`,
    restricted to nodes that already have the embedding property `prop`.

    Returns canonical (s, t) with s < t.
    """
    k = max(0, int(k))
    if k == 0:
        _dbg_iter("_sample_negative_pairs.skip", dbname=dbname, k=k)
        return []
    oversample = max(k * 3, 50)
    _dbg_iter("_sample_negative_pairs.start", dbname=dbname, prop=prop, k=k, oversample=oversample)

    cypher = f"""
    // (1) restrict to nodes with the embedding property
    MATCH (n)
    WHERE n.{prop} IS NOT NULL
    WITH collect(elementId(n)) AS ids, size(collect(n)) AS N

    // (2) sample candidate pairs (by elementId) via rand()
    CALL {{
      WITH ids, N
      UNWIND range(1, $oversample) AS _
      WITH ids, N,
           ids[toInteger(rand() * N)] AS a,
           ids[toInteger(rand() * N)] AS b
      WHERE a <> b
      WITH CASE WHEN a < b THEN a ELSE b END AS s,
           CASE WHEN a < b THEN b ELSE a END AS t
      RETURN DISTINCT s, t
      LIMIT $oversample
    }}

    // (3) keep only those with no edge of any type between them
    WITH s, t
    WHERE NOT EXISTS {{
      MATCH (x)--(y)
      WHERE (elementId(x) = s AND elementId(y) = t)
         OR (elementId(x) = t AND elementId(y) = s)
    }}

    RETURN DISTINCT s, t
    LIMIT $k
    """

    rows = run_data(
        cypher,
        {"k": k, "oversample": oversample},
        db=dbname,
    )
    pairs = [(r["s"], r["t"]) for r in rows]
    _dbg_iter("_sample_negative_pairs.done", dbname=dbname, prop=prop, requested=k, returned=len(pairs))
    return pairs


def _build_train_matrices(
    dbname: str,
    prop: str,
    pos_pairs: List[Tuple[str, str]],
    emb: Dict[str, np.ndarray],
    negative_ratio: float,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    STEP 3: Build X (features) and y (labels) for a training set.
      • positives from trainGraph
      • negatives from *non-edges* in the predicted DB
    """
    _dbg_iter(
        "_build_train_matrices.start",
        dbname=dbname,
        prop=prop,
        pos_pairs=len(pos_pairs),
        negative_ratio=negative_ratio,
    )
    # Positive features
    X_pos: List[np.ndarray] = []
    for s, t in pos_pairs:
        es = emb.get(s)
        et = emb.get(t)
        if es is None or et is None:
            continue
        X_pos.append(_hadamard(es, et))
    cnt_pos = len(X_pos)

    if cnt_pos == 0:
        _dbg_iter("_build_train_matrices.no_pos_after_emb", dbname=dbname, prop=prop)
        return np.zeros((0, 1)), np.zeros((0,), dtype=int), 0, 0

    want_neg = int(math.ceil(cnt_pos * max(0.0, min(10.0, float(negative_ratio)))))
    neg_pairs = _sample_negative_pairs(dbname, prop, want_neg)

    # Negative features
    X_neg: List[np.ndarray] = []
    for s, t in neg_pairs:
        es = emb.get(s)
        et = emb.get(t)
        if es is None or et is None:
            continue
        X_neg.append(_hadamard(es, et))
    cnt_neg = len(X_neg)

    if not X_neg:
        # No negatives ⇒ logistic regression cannot train properly
        _dbg_iter("_build_train_matrices.no_neg", dbname=dbname, prop=prop, cnt_pos=cnt_pos)
        return np.zeros((0, 1)), np.zeros((0,), dtype=int), cnt_pos, 0

    X = np.vstack([X_pos, X_neg])
    y = np.hstack([
        np.ones(cnt_pos, dtype=int),
        np.zeros(cnt_neg, dtype=int),
    ])

    _dbg_iter(
        "_build_train_matrices.done",
        dbname=dbname,
        prop=prop,
        cnt_pos=cnt_pos,
        cnt_neg=cnt_neg,
        X_shape=X.shape,
    )
    return X, y, cnt_pos, cnt_neg


def _predict_new_edges_for_iteration(
    dbname: str,
    prop: str,
    emb: Dict[str, np.ndarray],
    model: LogisticRegression,
    k: int,
    candidate_multiplier: int,
    threshold: float,
) -> Tuple[List[Tuple[str, str, float]], int, int]:
    """
    STEP 4 & 5:
      • Sample candidate non-edges from the predicted DB
      • Score them
      • Filter by threshold
      • Take top-K (falling back to best K even if <K above threshold)

    Returns:
      (selected_triples, total_scored, count_above_threshold)
    """
    want = max(1, int(k))
    mult = max(2, int(candidate_multiplier))
    thr = float(threshold)
    pool_size = max(want * mult, want + 5)
    _dbg_iter(
        "_predict_new_edges_for_iteration.start",
        dbname=dbname,
        prop=prop,
        want=want,
        mult=mult,
        thr=thr,
        pool_size=pool_size,
    )

    cand_pairs = _sample_negative_pairs(dbname, prop, pool_size)
    scored: List[Tuple[str, str, float]] = []
    for s, t in cand_pairs:
        es = emb.get(s)
        et = emb.get(t)
        if es is None or et is None:
            continue
        x = _hadamard(es, et).reshape(1, -1)
        p = float(model.predict_proba(x)[0, 1])
        scored.append((s, t, p))

    total_scored = len(scored)
    _dbg_iter("_predict_new_edges_for_iteration.scored", dbname=dbname, total_scored=total_scored)

    scored.sort(key=lambda x: x[2], reverse=True)

    above = [(s, t, p) for (s, t, p) in scored if p >= thr]
    count_above = len(above)

    if count_above >= want:
        selected = above[:want]
    else:
        selected = scored[:want]

    _dbg_iter(
        "_predict_new_edges_for_iteration.done",
        dbname=dbname,
        selected=len(selected),
        total_scored=total_scored,
        count_above=count_above,
    )
    return selected, total_scored, count_above

def _write_predicted_edges_inplace(
    dbname: str,
    triples: List[Tuple[str, str, float]],
    level: int,
    ts: int,
) -> int:
    """
    STEP 6:
      Write new predicted edges back into `dbname` as a single
      :PREDICTED relationship per undirected pair.

      We rely on the fact that (s, t) are already canonical by elementId
      (s < t), so we just create (s)-[:PREDICTED]->(t).

      • ON CREATE: set (predicted = level, timestamp = ts, probability = p)
      • ON MATCH: keep existing properties (i.e. don't clobber older levels)
    """
    if not triples:
        _dbg_iter("_write_predicted_edges_inplace.skip_empty", dbname=dbname)
        return 0

    _dbg_iter(
        "_write_predicted_edges_inplace.start",
        dbname=dbname,
        level=level,
        ts=ts,
        triples=len(triples),
    )

    rows = run_data(
        """
        UNWIND $triples AS row
        MATCH (s) WHERE elementId(s) = row.s
        MATCH (t) WHERE elementId(t) = row.t

        // We assume row.s < row.t by elementId (canonical pair),
        // and we only store the forward direction s -> t.
        CALL apoc.merge.relationship(
          s,
          'PREDICTED',
          {},
          {predicted: $level, timestamp: $ts, probability: row.p},
          t
        ) YIELD rel AS r

        RETURN count(*) AS relCount
        """,
        {
            "triples": [{"s": s, "t": t, "p": float(p)} for (s, t, p) in triples],
            "level": int(level),
            "ts": int(ts),
        },
        db=dbname,
    )
    rel_count = int(rows[0]["relCount"]) if rows else 0
    _dbg_iter("_write_predicted_edges_inplace.done", dbname=dbname, relCount=rel_count)
    return rel_count


# ----------------- Main iterative endpoint -----------------

@router.post("/iterate")
def iterate_prediction(body: IterateBody) -> Dict[str, Any]:
    """
    Iterative LP pipeline for a *predicted* DB:

      STEP 1: resolve which predicted DB we are going to work on
      STEP 2: run the splitter inside that predicted DB → trainGraph / testGraph
      STEP 3: recompute node embeddings on trainGraph (FastRP / Node2Vec)
      STEP 4: get positive edges from trainGraph, build pos+neg features
      STEP 5: sample candidate non-edges, score, filter by threshold, select top-K
      STEP 6: write new predicted edges back into chosen_db with level & timestamp
    """
    _dbg_iter("iterate_prediction.enter", body=body.dict())

    fam = body.embeddingFamily.strip()
    if fam not in EMB_FAMILIES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown embeddingFamily '{fam}'. Choose one of {EMB_FAMILIES}.",
        )

    # --- STEP 1: resolve target predicted DB based on current base DB + family ---
    base = current_database()
    target_db = _predicted_db_name(base, fam)
    exists = _db_exists(target_db)

    _dbg_iter(
        "iterate_prediction.resolved",
        base=base,
        fam=fam,
        target_db=target_db,
        exists=exists,
    )

    if not exists:
        # We cannot iterate if the base predicted DB doesn’t exist yet
        raise HTTPException(
            status_code=404,
            detail=(
                f"Predicted DB '{target_db}' not found. "
                f"Run the initial LP for {fam} first."
            ),
        )

    # --- STEP 2: run the splitter INSIDE the chosen predicted DB ---
    with _with_db_bound(split_mod, target_db):
        # Temporarily restrict feature types to numeric/bool for GDS safety
        _orig_allowed = getattr(
            split_mod,
            "_ALLOWED_SCALARS",
            (int, float, bool, str),
        )
        split_mod._ALLOWED_SCALARS = (int, float, bool)
        try:
            # Clear feature cache for this DB so it rescans properties
            split_mod.clear_feature_cache(db=target_db)

            # Run the master split – creates trainGraph / testGraph in GDS
            split_result = split_mod.execute_master_split(
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

            # Make sure result is JSON-serializable for FastAPI response
            if hasattr(split_result, "model_dump"):
                split_payload = split_result.model_dump()
            elif hasattr(split_result, "dict"):
                split_payload = split_result.dict()
            else:
                split_payload = split_result
        finally:
            # Restore original scalar whitelist no matter what
            split_mod._ALLOWED_SCALARS = _orig_allowed

    # Current max predicted level & timestamp *before* this iteration
    prev_level, prev_ts = _compute_level_and_ts(target_db)

    # --- STEP 3: new embeddings on trainGraph in the chosen predicted DB ---
    _assert_sklearn_iter()
    prop = _sanitize_prop(body.embeddingProperty)
    emb_info = _compute_embeddings_for_family(
        dbname=target_db,
        family=fam,
        prop=prop,
        graph_name="trainGraph",
    )

    # --- STEP 4: positive pairs from trainGraph ---
    pos_pairs = _get_train_positive_pairs(target_db, "trainGraph")

    # load embeddings from the predicted DB
    emb = _load_embeddings_from_db(target_db, prop)
    if not emb:
        _dbg_iter("iterate_prediction.no_embeddings", dbname=target_db, prop=prop)
        raise HTTPException(
            status_code=400,
            detail=f"No nodes found with embedding property '{prop}' in '{target_db}'.",
        )

    # --- STEP 5: build features + train logistic regression ---
    Xtr, ytr, npos_tr, nneg_tr = _build_train_matrices(
        dbname=target_db,
        prop=prop,
        pos_pairs=pos_pairs,
        emb=emb,
        negative_ratio=body.negativeRatio,
    )

    if Xtr.shape[0] == 0 or npos_tr == 0 or nneg_tr == 0:
        _dbg_iter(
            "iterate_prediction.no_training_pairs",
            dbname=target_db,
            X_shape=Xtr.shape,
            npos_tr=npos_tr,
            nneg_tr=nneg_tr,
        )
        raise HTTPException(
            status_code=400,
            detail=(
                "No usable training pairs for iterative LP "
                "(check embeddings exist for endpoints and at least one negative sample)."
            ),
        )

    model = LogisticRegression(max_iter=200, n_jobs=None)
    model.fit(Xtr, ytr)
    _dbg_iter(
        "iterate_prediction.model_trained",
        dbname=target_db,
        npos_tr=npos_tr,
        nneg_tr=nneg_tr,
    )

    # --- STEP 6: sample candidate non-edges & predict new edges ---
    selected_triples, total_scored, count_above = _predict_new_edges_for_iteration(
        dbname=target_db,
        prop=prop,
        emb=emb,
        model=model,
        k=body.predictK,
        candidate_multiplier=body.candidateMultiplier,
        threshold=body.probThreshold,
    )

    new_level = prev_level + 1
    new_ts = prev_ts + 1

    rel_written = _write_predicted_edges_inplace(
        dbname=target_db,
        triples=selected_triples,
        level=new_level,
        ts=new_ts,
    )

    _dbg_iter(
        "iterate_prediction.completed",
        base=base,
        fam=fam,
        target_db=target_db,
        prev_level=prev_level,
        new_level=new_level,
        prev_ts=prev_ts,
        new_ts=new_ts,
        selected=len(selected_triples),
        rel_written=rel_written,
    )

    # Final JSON response
    return {
        "ok": True,
        "baseDatabase": base,
        "embeddingFamily": fam,
        "targetPredictedDb": target_db,
        "targetExists": True,
        "embeddingProperty": prop,
        "splitResult": split_payload,
        "prevPredictedLevel": prev_level,
        "prevTimestamp": prev_ts,
        "newPredictedLevel": new_level,
        "newTimestamp": new_ts,
        "trainPairs": npos_tr,
        "negTrainPairs": nneg_tr,
        "embeddingInfo": emb_info,
        "prediction": {
            "requestedK": body.predictK,
            "candidateMultiplier": body.candidateMultiplier,
            "probThreshold": float(body.probThreshold),
            "totalCandidatesScored": total_scored,
            "candidatesAboveThreshold": count_above,
            "selectedNewPairs": len(selected_triples),
            "relationshipsWritten": rel_written,
        },
    }
