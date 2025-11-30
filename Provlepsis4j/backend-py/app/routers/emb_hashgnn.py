# app/routers/emb_hashgnn.py
from fastapi import APIRouter, Body, HTTPException
from typing import Any, Dict, List, Optional
from math import ceil
from ..db import run_data
from .gds_context import resolve_graph_name

router = APIRouter()

# ---------- General helpers ----------

def _sanitize_prop(name: str) -> str:
    return "".join(ch for ch in str(name) if ch.isalnum() or ch == "_") or "embedding_hash_128"

def _as_int_ge1(val: Any, default: int) -> int:
    try:
        num = float(val)
    except (TypeError, ValueError):
        return max(1, int(default))
    return max(1, int(ceil(num)))

# ---------- Discovery / filtering (HashGNN-specific) ----------

# We only exclude "id" (NOT "label")
ALWAYS_EXCLUDE = {"id"}

# We still avoid passing previously written embedding-like props by name/prefix
EMBEDDING_PREFIXES = ["fastrp_", "node2vec_", "graphsage_", "hashgnn_", "embedding_"]
EMBEDDING_EXACTS = {"FastRP", "Node2Vec", "GraphSAGE", "HashGNN"}

# Allowed scalars for quick sanity checks in “supported” probes
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

def _exclude_embeddingish_props(props: List[str]) -> List[str]:
    out: List[str] = []
    for p in props:
        if p in EMBEDDING_EXACTS:
            continue
        if any(p.startswith(pref) for pref in EMBEDDING_PREFIXES):
            continue
        out.append(p)
    return out

def _filter_supported_props(props: List[str]) -> List[str]:
    """
    Drop properties that obviously aren't numeric / numeric-list / scalar-ish (same spirit as FastRP).
    """
    safe: List[str] = []
    for p in props:
        rows = run_data(
            "WITH $k AS k MATCH (n) WHERE n[k] IS NOT NULL RETURN n[k] AS v LIMIT 500",
            {"k": p},
        )
        # Keep if no samples (could be sparse/nullable and still valid)
        if not rows:
            safe.append(p)
            continue
        ok = True
        for r in rows:
            if not _looks_supported_value(r.get("v")):
                ok = False
                break
        if ok:
            safe.append(p)
    return safe

def _is_binary_value(v) -> bool:
    """
    HashGNN (without binarize) requires that every non-null value is 0 or 1.
    """
    if v is None:
        return True
    if isinstance(v, (int, float)):
        return v in (0, 1)
    if isinstance(v, list):
        for x in v:
            if x is None:
                continue
            if not isinstance(x, (int, float)) or x not in (0, 1):
                return False
        return True
    return False

def _filter_binary_props(props: List[str]) -> List[str]:
    """
    Keep only props whose sampled values are all binary (0/1), ignoring nulls.
    """
    out: List[str] = []
    for p in props:
        if p in ALWAYS_EXCLUDE:
            continue
        rows = run_data(
            "WITH $k AS k "
            "MATCH (n) WHERE n[k] IS NOT NULL "
            "RETURN n[k] AS v LIMIT 500",
            {"k": p},
        )
        if not rows:
            # All null in sample; treat as OK
            out.append(p)
            continue
        ok = True
        for r in rows:
            if not _is_binary_value(r.get("v")):
                ok = False
                break
        if ok:
            out.append(p)
    return out

def _validate_props_exist_in_projection(graph: str, props: List[str]) -> List[str]:
    """
    Return only those props that are actually present in the in-memory GDS projection.
    """
    if not props:
        return []
    rows = run_data(
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
    )
    return rows[0]["okProps"] if rows else []

def _discover_candidate_props(graph: str) -> List[str]:
    """
    Start from projected node keys (same spirit as the FastRP discovery),
    then exclude embedding-y names and 'id'.
    """
    rows = run_data(
        """
        WITH $g AS g
        CALL gds.degree.stream(g) YIELD nodeId
        WITH g, nodeId LIMIT 1
        WITH g, gds.util.asNode(nodeId) AS n
        RETURN [p IN keys(n)
                WHERE NOT p IN ['FastRP','Node2Vec','GraphSAGE','HashGNN']
                  AND NOT p STARTS WITH 'embedding_'
               ] AS projectedProps
        """,
        {"g": graph},
    )
    props = rows[0]["projectedProps"] if rows else []
    props = [p for p in props if p not in ALWAYS_EXCLUDE]  # exclude only 'id'
    props = _exclude_embeddingish_props(props)
    return props

def _discover_hashgnn_features(graph: str) -> List[str]:
    """
    HashGNN-specific discovery: only keep props that are present in the projection,
    look supported (scalar/list numerics/booleans/strings), and are BINARY.
    """
    props = _discover_candidate_props(graph)
    props = _filter_supported_props(props)
    props = _validate_props_exist_in_projection(graph, props)
    props = _filter_binary_props(props)
    return props

# ---------- Cypher builders ----------

def _stmt_with_features(write_prop: str, use_binarize: bool) -> str:
    """
    Build a call that uses given featureProperties. Optionally include binarizeFeatures.
    """
    base = [
        "CALL gds.hashgnn.stream($g, {",
        "  iterations: $it,",
        "  outputDimension: $dim,",
        "  embeddingDensity: $dens,",
        "  featureProperties: $feats",
    ]
    if use_binarize:
        base.append(",  binarizeFeatures: true")
    base.append("}) YIELD nodeId, embedding ")
    base.append("WITH gds.util.asNode(nodeId) AS n, [x IN embedding | toFloat(x)] AS v ")
    base.append(f"SET n.{write_prop} = v ")
    base.append("RETURN count(*) AS wrote")
    return "".join(base)

def _stmt_generate_only(write_prop: str) -> str:
    """
    Build a call that ignores features and generates synthetic ones.
    """
    return (
        "CALL gds.hashgnn.stream($g, {"
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

# ---------- Route ----------

@router.post("/write")
def hashgnn_write(body: Dict[str, Any] = Body(default={})):
    g    = "trainGraph"
    dim  = _as_int_ge1(body.get("outputDimension", body.get("embeddingDimension", 128)), 128)
    dens = _as_int_ge1(body.get("embeddingDensity", 10), 10)
    it   = _as_int_ge1(body.get("iterations", 10), 10)
    prop = _sanitize_prop(body.get("writeProperty", "embedding_hash_128"))

    # Accept explicit featureProperties or discover automatically
    requested: Optional[List[str]] = body.get("featureProperties")
    if requested is not None:
        # Respect user list, but validate against projection and HashGNN's binary requirement
        requested = [p for p in requested if p not in ALWAYS_EXCLUDE]  # exclude 'id' only
        feats = _validate_props_exist_in_projection(g, requested)
        feats = _filter_binary_props(feats)
    else:
        feats = _discover_hashgnn_features(g)

    params = {"g": g, "it": it, "dim": dim, "dens": dens}
    used_features: List[str] = feats
    fallback_reason: Optional[str] = None

    # Branching:
    # 1) If we have features, try with binarizeFeatures=true. If server rejects that option,
    #    retry without it (features are already binary after our filter).
    # 2) If no valid features remain, generateFeatures fallback.
    try:
        if used_features:
            cypher = _stmt_with_features(prop, use_binarize=True)
            rows = run_data(cypher, {**params, "feats": used_features})
        else:
            cypher = _stmt_generate_only(prop)
            rows = run_data(cypher, params)
            fallback_reason = "No binary featureProperties found; used generateFeatures instead."
    except Exception as e1:
        # Retry path:
        # - If the failure was likely due to unknown 'binarizeFeatures' or a similar config issue,
        #   try again without binarizeFeatures (features are binary already).
        if used_features:
            try:
                cypher = _stmt_with_features(prop, use_binarize=False)
                rows = run_data(cypher, {**params, "feats": used_features})
                fallback_reason = "binarizeFeatures not supported by this GDS build; used raw binary features."
            except Exception as e2:
                # Last fallback: generate features
                cypher = _stmt_generate_only(prop)
                rows = run_data(cypher, params)
                fallback_reason = (
                    "Using featureProperties failed; fell back to generateFeatures. "
                    f"First error: {str(e1)} | Second error: {str(e2)}"
                )
                used_features = []
        else:
            # We were already in generate mode; just raise a clean error
            raise HTTPException(status_code=400, detail=str(e1))

    wrote = rows[0]["wrote"] if rows else 0
    return {
        "ok": True,
        "graphName": g,
        "writeProperty": prop,
        "wrote": wrote,
        "usedFeatures": used_features,
        "config": {
            "iterations": it,
            "outputDimension": dim,
            "embeddingDensity": dens,
            **({"featureProperties": used_features} if used_features else {
                "generateFeatures": {"dimension": dim, "densityLevel": dens}
            })
        },
        "fallbackReason": fallback_reason,
    }

@router.post("/trainWrite")
def hashgnn_train_write(body: Dict[str, Any] = Body(default={})):
    # alias, for symmetry with the other routers
    return hashgnn_write(body)
