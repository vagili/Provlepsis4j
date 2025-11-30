# app/routers/emb_fastrp.py
from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from ..db import run_data
from neo4j.exceptions import Neo4jError

router = APIRouter()

# ---- Configuration / helpers ----

# Keep in sync with split.py exclusions (these are only for *features* filtering)
EMBEDDING_PREFIXES = ["fastrp_", "node2vec_", "graphsage_", "hashgnn_"]
EMBEDDING_EXACTS = {"FastRP", "Node2Vec", "GraphSAGE", "HashGNN"}
INTERNAL_KEYS = {"_id", "_tmp", "_ts", "id", "ID"}

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

def _filter_supported_props(props: List[str]) -> List[str]:
    safe: List[str] = []
    for p in props:
        rows = run_data(
            "WITH $k AS k MATCH (n) WHERE n[k] IS NOT NULL RETURN n[k] AS v",
            {"k": p},
        )
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

def _exclude_embedding_props(props: List[str]) -> List[str]:
    out: List[str] = []
    for p in props:
        if p in INTERNAL_KEYS or p in EMBEDDING_EXACTS:
            continue
        if any(p.startswith(pref) for pref in EMBEDDING_PREFIXES):
            continue
        out.append(p)
    return out

def _validate_props_exist_in_projection(graph: str, props: List[str]) -> List[str]:
    """
    Return only those props that are actually present in the in-memory GDS projection.
    We probe each key in a single Cypher using an UNWIND + subquery.
    If a key isn't projected, the subquery won't yield a row (no exception leaks out).
    """
    if not props:
        return []

    rows = run_data("""
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
    """, {"g": graph, "props": props})
    return rows[0]["okProps"] if rows else []


def _discover_feature_props(graph: str) -> List[str]:
    # Pull a single projected node’s keys and keep only numeric-ish props (no APOC needed)
    rows = run_data("""
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
    """, {"g": graph})
    props = rows[0]["numericProjectedProps"] if rows else []

    # Extra safety: exclude embedding-y names, type-check samples, and ensure they exist in projection
    props = _exclude_embedding_props(props)
    props = _filter_supported_props(props)
    props = _validate_props_exist_in_projection(graph, props)
    return props

# ---- Request & route ----

class FastRPWriteBody(BaseModel):
    graphName: str = "trainGraph"
    embeddingDimension: int = Field(default=128, ge=2, le=2048)
    writeProperty: str = "FastRP"                # <— write as 'FastRP'
    propertyRatio: float = Field(default=0.0, ge=0.0, le=1.0)
    featureProperties: Optional[List[str]] = None  # optional manual override

@router.post("/write")
def fastrp_write(body: FastRPWriteBody):
    g = body.graphName
    dim = int(body.embeddingDimension)
    wp = body.writeProperty
    ratio = float(body.propertyRatio)

    if body.featureProperties is not None:
        features = _validate_props_exist_in_projection(g, _exclude_embedding_props(body.featureProperties))
        used_features = features
    else:
        used_features = _discover_feature_props(g)

    params: Dict[str, Any] = {
        "g": g,
        "dim": dim,
        "wp": wp,
        "ratio": ratio,
        "features": used_features,
    }

    if ratio > 0 and not used_features:
        return {
            "ok": False,
            "graphName": g,
            "procedure": "gds.fastRP.write",
            "error": "When propertyRatio > 0, no projected feature properties were found. "
                     "Either set propertyRatio to 0, or pass featureProperties explicitly."
        }

    if ratio > 0:
        cypher = """
        CALL gds.fastRP.write($g, {
          embeddingDimension: $dim,
          writeProperty: $wp,
          propertyRatio: $ratio,
          featureProperties: $features
        })
        YIELD nodePropertiesWritten
        RETURN nodePropertiesWritten AS written
        """
    else:
        cypher = """
        CALL gds.fastRP.write($g, {
          embeddingDimension: $dim,
          writeProperty: $wp
        })
        YIELD nodePropertiesWritten
        RETURN nodePropertiesWritten AS written
        """

    try:
        rows = run_data(cypher, params)
        written = int(rows[0]["written"]) if rows else 0
        return {
            "ok": True,
            "graphName": g,
            "procedure": "gds.fastRP.write",
            "writeProperty": wp,
            "propertyRatio": ratio,
            "embeddingDimension": dim,
            "usedFeatures": used_features,
            "nodePropertiesWritten": written
        }
    except Neo4jError as e:
        return {
            "ok": False,
            "graphName": g,
            "procedure": "gds.fastRP.write",
            "error": f"{e.code}: {e.message}"
        }
    except Exception as e:
        return {
            "ok": False,
            "graphName": g,
            "procedure": "gds.fastRP.write",
            "error": str(e)
        }
