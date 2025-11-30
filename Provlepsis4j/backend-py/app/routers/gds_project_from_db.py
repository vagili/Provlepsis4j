# app/routers/gds_project_from_db.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Literal, Dict, Any

from ..db import run_data, run, current_database

router = APIRouter()

class ProjectFromDBBody(BaseModel):
    name: str                              # projection name to create
    nodeLabels: Optional[List[str]] = None # if omitted -> use all labels
    relationshipTypes: Optional[List[str]] = None # if omitted -> use all rel types
    orientation: Literal["UNDIRECTED", "NATURAL", "REVERSE"] = "UNDIRECTED"
    dropIfExists: bool = True              # drop any existing projection with same name

@router.get("/db/meta")
def db_meta() -> Dict[str, Any]:
    labels = [r["label"] for r in run_data(
        "CALL db.labels() YIELD label RETURN label ORDER BY label"
    )]

    rels = [r["type"] for r in run_data(
        "CALL db.relationshipTypes() YIELD relationshipType "
        "RETURN relationshipType AS type ORDER BY type"
    )]

    nodeCount = run_data("MATCH (n) RETURN count(n) AS c")[0]["c"]
    relCount  = run_data("MATCH ()-[r]->() RETURN count(r) AS c")[0]["c"]

    return {
        "database": current_database(),
        "nodeCount": int(nodeCount),
        "relationshipCount": int(relCount),
        "labels": labels,
        "relationshipTypes": rels,
    }

@router.post("/gds/project-from-db")
def project_from_db(body: ProjectFromDBBody):
    # Resolve labels / relationship types if not provided
    labels = body.nodeLabels or [
        r["label"] for r in run_data("CALL db.labels() YIELD label RETURN label")
    ]
    rel_types = body.relationshipTypes or [
        r["type"] for r in run_data(
            "CALL db.relationshipTypes() YIELD relationshipType "
            "RETURN relationshipType AS type"
        )
    ]

    # Optionally drop any existing projection (ignore if it doesn't exist)
    if body.dropIfExists:
        try:
            run("CALL gds.graph.drop($name, false)", {"name": body.name})
        except Exception:
            pass

    # Build relationship config map
    rel_cfg = {t: {"orientation": body.orientation} for t in rel_types}

    # Create the projection; don't return the raw row to the client
    try:
        # Either call without YIELD…
        run("CALL gds.graph.project($name, $labels, $rels)",
            {"name": body.name, "labels": labels, "rels": rel_cfg})
        # …or with YIELD but ignore the row to avoid JSON encoding issues:
        # run_data("CALL gds.graph.project($name, $labels, $rels) "
        #          "YIELD graphName RETURN graphName",
        #          {"name": body.name, "labels": labels, "rels": rel_cfg})
    except Exception as e:
        # Surface clear error to the UI
        raise HTTPException(status_code=400, detail=f"Projection failed: {e}")

    # Keep the response dead-simple; the UI re-fetches the list anyway.
    return {"ok": True, "graphName": body.name}
