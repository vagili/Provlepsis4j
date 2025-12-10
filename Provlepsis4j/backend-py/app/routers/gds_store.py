# app/routers/gds_store.py
from fastapi import APIRouter, HTTPException, Body
from typing import List, Optional, Dict, Any

from ..db import run, run_data
from .gds_context import set_graph_context, GraphCtxIn

router = APIRouter()

# ---- Optional feature helpers --------------------------------
try:
    from . import feature as _feature_mod
    _FEATURES_OK = True
except Exception:
    _feature_mod = None
    _FEATURES_OK = False

# Embedding properties to exclude if we compute props locally
_EXCLUDE_KEYS = ["Node2Vec", "FastRP", "GraphSAGE", "HashGNN"]

# ---- Local discovery of full-coverage STRING keys (pure Cypher) --------------
def _discover_string_keys_full_coverage(coverage: float = 1.0) -> List[str]:
    """
    Keys present & non-null on at least coverage fraction of nodes,
    whose present values are all non-numeric strings.
    Excludes embedding writeProperties and keys already starting with numeric_.
    """
    cypher = """
    MATCH (n) WITH count(n) AS N
    CALL db.propertyKeys() YIELD propertyKey AS k
    WITH N, k
    WHERE NOT k IN $exclude AND NOT k STARTS WITH 'numeric_'
    MATCH (m) WHERE m[k] IS NOT NULL
    WITH N, k,
         count(m) AS present,
         sum(CASE WHEN toFloatOrNull(m[k]) IS NULL THEN 1 ELSE 0 END) AS string_ok,
         toInteger(ceil($coverage * N)) AS cov_min
    WHERE present >= cov_min AND string_ok = present
    RETURN collect(k) AS keys
    """
    rows = run_data(cypher, {"exclude": _EXCLUDE_KEYS, "coverage": coverage})
    return rows[0]["keys"] if rows else []

# ---- Local one-hot writer for a single key (pure Cypher) ---------------------
def _local_onehot_write_numeric_key(key: str, max_categories: int = 256) -> Dict[str, Any]:
    """
    Creates numeric_<key> with one-hot FLOAT[] values based on the scalar string categories of `key`.
    Requires key to be full-coverage and pure-string (call _discover_string_keys_full_coverage first).
    """
    # 1) Collect categories (deterministic order)
    rows = run_data(
        """
        WITH $k AS k
        MATCH (n)
        WITH n[k] AS v
        WHERE v IS NOT NULL
        WITH v ORDER BY v
        RETURN collect(DISTINCT v) AS cats
        """,
        {"k": key},
    )
    cats = rows[0]["cats"] if rows else []
    if not cats:
        return {"key": key, "status": "skipped-empty-categories"}

    if len(cats) > max_categories:
        return {
            "key": key,
            "status": "skipped-too-many-categories",
            "numCategories": len(cats),
        }

    # 2) Write numeric_<key> using literal property name (no dynamic SET)
    # Note: We deliberately DO NOT overwrite the original key; we write a new property.
    out_prop = f"numeric_{key}"
    cypher = f"""
    WITH $cats AS cats
    MATCH (n)
    WITH n, cats, n.`{key}` AS v
    SET n.`{out_prop}` = [i IN range(0, size(cats)-1) |
                            CASE WHEN cats[i] = v THEN 1.0 ELSE 0.0 END]
    RETURN count(n) AS written
    """
    w = run_data(cypher, {"cats": cats})
    return {
        "key": key,
        "numericKey": out_prop,
        "status": "converted",
        "numCategories": len(cats),
        "nodesWritten": int(w[0]["written"]) if w else 0,
    }

# ---- Local computation of projection props (numeric + numeric_*) -------------
def _compute_projection_props_locally(coverage: float = 1.0) -> List[str]:
    """
    Includes numeric scalars and 'numeric_*' properties with required coverage.
    Excludes embedding writeProperties.
    """
    cypher = """
    MATCH (n) WITH count(n) AS N
    CALL db.propertyKeys() YIELD propertyKey AS k
    WITH N, k
    WHERE NOT k IN $exclude
    MATCH (m) WHERE m[k] IS NOT NULL
    WITH N, k,
         count(m) AS present,
         sum(CASE WHEN toFloatOrNull(m[k]) IS NOT NULL THEN 1 ELSE 0 END) AS numeric_ok,
         toInteger(ceil($coverage * N)) AS cov_min
    WHERE present >= cov_min
      AND (numeric_ok = present OR k STARTS WITH 'numeric_')
    RETURN collect(k) AS props
    """
    rows = run_data(cypher, {"exclude": _EXCLUDE_KEYS, "coverage": coverage})
    props = rows[0]["props"] if rows else []
    return sorted([p for p in props if p not in _EXCLUDE_KEYS])

# ---- Auto pipeline: try helpers; else do local conversion + selection --------
def _auto_prepare_features(coverage: float = 1.0, max_categories: int = 256) -> Dict[str, Any]:
    """
    Runs conversion + selection and returns:
      { props: [...], path: "feature_module" | "local", conversions: [...] }
    """
    # Try feature module first
    if _FEATURES_OK and _feature_mod is not None:
        conversions: List[Dict[str, Any]] = []
        path = "feature_module"
        try:
            # 1) conversion (idempotent)
            BulkBody = getattr(_feature_mod, "BulkConvertBody", None)
            if BulkBody is not None:
                _feature_mod.strings_to_numeric_bulk(BulkBody(coverage=coverage, max_categories=max_categories))
            else:
                _feature_mod.strings_to_numeric_bulk({"coverage": coverage, "max_categories": max_categories})
        except Exception as e:
            # fall back to local path if module conversion fails
            path = f"feature_module_failed: {e}"
        else:
            # 2) final list
            try:
                if hasattr(_feature_mod, "get_projection_props"):
                    props = _feature_mod.get_projection_props()
                    return {
                        "props": sorted([p for p in (props or []) if p not in _EXCLUDE_KEYS]),
                        "path": path,
                        "conversions": conversions,
                    }
            except Exception as e:
                path = f"feature_module_get_list_failed: {e}"

    # Local path: discover full-coverage string keys; convert; compute props
    path = "local"
    conversions: List[Dict[str, Any]] = []

    try:
        string_keys = _discover_string_keys_full_coverage(coverage=coverage)
        for k in string_keys:
            conversions.append(_local_onehot_write_numeric_key(k, max_categories=max_categories))
    except Exception as e:
        conversions.append({"status": "local-conversion-error", "detail": str(e)})

    try:
        props = _compute_projection_props_locally(coverage=coverage)
    except Exception as e:
        props = []
        conversions.append({"status": "local-selection-error", "detail": str(e)})

    return {"props": props, "path": path, "conversions": conversions}

# ---- Routes ------------------------------------------------------------------

@router.get("/store/summary")
def store_summary():
    rows = run_data(
        "MATCH (n) WITH count(n) AS nodes "
        "MATCH ()-[r]->() WITH nodes, count(r) AS relationships "
        "RETURN nodes, relationships"
    )
    row = rows[0] if rows else {"nodes": 0, "relationships": 0}
    return {
        "nodes": int(row.get("nodes", 0)),
        "relationships": int(row.get("relationships", 0)),
    }

@router.post("/graph/project-store")
def project_store_graph(
    body: Dict[str, Any] = Body(
        default={"name": "fullGraph"},
        examples={"default": {"name": "fullGraph", "orientation": "UNDIRECTED"}},
    )
):
    """
    Project the current DB into a GDS in-memory graph with the given name.

    Body (optional):
      - name: str (default "fullGraph")
      - orientation: "UNDIRECTED" | "NATURAL" | "REVERSE" (default "UNDIRECTED")
      - labels: [str] (if omitted, project all labels)
      - relationshipTypes: [str] (if omitted, project all relationship types)
      - autoEncodeStrings: bool (default True) -> run string->numeric pipeline before projection
      - nodeProperties: [str] (optional override, wins over auto-discovered list)
      - maxCategories: int (default 256) for one-hot
    """
    name = str(body.get("name") or "fullGraph").strip()
    if not name:
        raise HTTPException(status_code=400, detail="Projection name is required.")

    orientation = str(body.get("orientation") or "UNDIRECTED").upper()
    if orientation not in {"UNDIRECTED", "NATURAL", "REVERSE"}:
        raise HTTPException(status_code=400, detail=f"Invalid orientation: {orientation}")

    auto_encode = body.get("autoEncodeStrings", True)
    max_categories = int(body.get("maxCategories") or 256)

    # Resolve labels / relationship types if not provided
    labels: Optional[List[str]] = body.get("labels")
    if labels is None:
        lr = run_data("CALL db.labels() YIELD label RETURN collect(label) AS labels")
        labels = lr[0]["labels"] if lr else []

    rel_types: Optional[List[str]] = body.get("relationshipTypes")
    if rel_types is None:
        rr = run_data(
            "CALL db.relationshipTypes() YIELD relationshipType "
            "RETURN collect(relationshipType) AS types"
        )
        rel_types = rr[0]["types"] if rr else []

    if not labels:
        raise HTTPException(status_code=400, detail="No labels found to project.")
    if not rel_types:
        raise HTTPException(status_code=400, detail="No relationship types found to project.")

    # Node properties to include
    conversion_path_info: str | None = None
    conversions_log: List[Dict[str, Any]] = []
    node_props: Optional[List[str]] = body.get("nodeProperties")

    if node_props is None:
        if auto_encode:
            result = _auto_prepare_features(coverage=1.0, max_categories=max_categories)
            node_props = result.get("props") or []
            conversion_path_info = result.get("path")
            conversions_log = result.get("conversions") or []
        else:
            node_props = _compute_projection_props_locally(coverage=1.0)

    # Relationship projection map
    rel_proj = {t: {"orientation": orientation} for t in rel_types}

    # Drop existing graph if present (ignore if absent)
    try:
        run("CALL gds.graph.drop($name, false)", {"name": name})
    except Exception:
        pass

    # Create projection (with nodeProperties if any)
    try:
        if node_props:
            run(
                "CALL gds.graph.project($name, $labels, $rels, {nodeProperties: $props})",
                {"name": name, "labels": labels, "rels": rel_proj, "props": node_props},
            )
        else:
            run("CALL gds.graph.project($name, $labels, $rels)",
                {"name": name, "labels": labels, "rels": rel_proj})
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Projection failed: {e}")

    # Fetch metadata
    meta = run_data(
        "CALL gds.graph.list($name) "
        "YIELD graphName, nodeCount, relationshipCount, creationTime "
        "RETURN graphName, nodeCount, relationshipCount, creationTime",
        {"name": name},
    )
    if meta:
        m = meta[0]
        graph_name = m.get("graphName", name)
        node_count = int(m.get("nodeCount", 0)) if m.get("nodeCount") is not None else None
        rel_count = int(m.get("relationshipCount", 0)) if m.get("relationshipCount") is not None else None
        creation_time = str(m.get("creationTime")) if m.get("creationTime") is not None else None
    else:
        graph_name = name
        node_count = rel_count = creation_time = None

    # Set active graph context so follow-up UI calls work
    set_graph_context(GraphCtxIn(graphName=graph_name))

    return {
        "ok": True,
        "graphName": graph_name,
        "nodeCount": node_count,
        "relationshipCount": rel_count,
        "creationTime": creation_time,
        "nodePropertiesUsed": node_props or [],
        "autoEncodeStrings": bool(auto_encode),
        "featuresModuleDetected": bool(_FEATURES_OK),
        "conversionPath": conversion_path_info,
        "conversions": conversions_log,
    }

@router.post("/graph/drop")
def drop_projected_graph(body: dict = Body(...)):
    name = (body.get("name") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="Missing 'name'")

    rows = run_data(
        "CALL gds.graph.drop($name, false) YIELD graphName RETURN graphName",
        {"name": name},
    )
    dropped = rows[0]["graphName"] if rows else name
    return {"dropped": dropped}