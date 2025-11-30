# app/routers/emb_graphsage.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from ..db import run, run_data
from .gds_context import resolve_graph_name

router = APIRouter()

class GraphSAGETrainWriteBody(BaseModel):
    graphName: Optional[str] = None
    modelName: str = "sage_model"
    writeProperty: str = "embedding_sage_64"
    featureProperties: Optional[List[str]] = None
    aggregator: str = "mean"
    sampleSizes: List[int] = [25, 10]
    embeddingDimension: int = Field(default=64, ge=2, le=2048)
    epochs: int = 5
    learningRate: float = 0.01

# ---- same filters as FastRP ----
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
            "WITH $k AS k MATCH (n) WHERE n[k] IS NOT NULL RETURN n[k] AS v LIMIT 500",
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
    props = _exclude_embedding_props(props)
    props = _filter_supported_props(props)
    props = _validate_props_exist_in_projection(graph, props)
    return props

def _resolve_graphsage_prefix() -> str:
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
        {"names": names}
    )
    present = set(rows[0]["present"] if rows else [])
    train = "gds.graphSage.train" if "gds.graphSage.train" in present else (
            "gds.beta.graphSage.train" if "gds.beta.graphSage.train" in present else None)
    write = "gds.graphSage.write" if "gds.graphSage.write" in present else (
            "gds.beta.graphSage.write" if "gds.beta.graphSage.write" in present else None)
    if not train or not write:
        raise HTTPException(
            status_code=400,
            detail=("GraphSAGE procedures not found. Your GDS version/edition may not include GraphSAGE "
                    "(look for gds.beta.graphSage.*).")
        )
    return train.rsplit(".", 1)[0]

def _drop_model_if_exists(name: str):
    # Best-effort drop; ignores "not found" errors across GDS versions.
    try:
        run("CALL gds.model.drop($name) YIELD modelName RETURN modelName", {"name": name})
    except Exception:
        pass

@router.post("/trainWrite")
def graphsage_train_write(body: GraphSAGETrainWriteBody):
    g = "trainGraph"
    prefix = _resolve_graphsage_prefix()

    if body.featureProperties is not None:
        requested = _exclude_embedding_props(body.featureProperties)
        used_features = _validate_props_exist_in_projection(g, requested)
    else:
        used_features = _discover_feature_props(g)

    if not used_features:
        raise HTTPException(
            status_code=400,
            detail=("No suitable numeric feature properties were found in the in-memory projection. "
                    "Either supply 'featureProperties' explicitly, or ensure your projected nodes "
                    "have numeric (or numeric-list) properties.")
        )

    # Drop any existing model with the same name to avoid conflicts
    _drop_model_if_exists(body.modelName)

    train_cfg: Dict[str, Any] = {
        "modelName": body.modelName,
        "featureProperties": used_features,
        "aggregator": body.aggregator,
        "sampleSizes": body.sampleSizes,
        "embeddingDimension": body.embeddingDimension,
        "epochs": body.epochs,
        "learningRate": body.learningRate,
    }
    # Train (no result needed)
    run(f"CALL {prefix}.train($g, $cfg)", {"g": g, "cfg": train_cfg})

    write_cfg: Dict[str, Any] = {
        "modelName": body.modelName,
        "writeProperty": body.writeProperty,
    }
    # Write and capture how many nodes were written
    rows = run_data(
        f"CALL {prefix}.write($g, $cfg) "
        f"YIELD nodePropertiesWritten "
        f"RETURN nodePropertiesWritten AS written",
        {"g": g, "cfg": write_cfg}
    )
    written = int(rows[0]["written"]) if rows else 0

    return {
        "ok": True,
        "graphName": g,
        "modelName": body.modelName,
        "writeProperty": body.writeProperty,
        "featureProperties": used_features,
        "usedProcedures": {"train": f"{prefix}.train", "write": f"{prefix}.write"},
        "nodePropertiesWritten": written
    }
