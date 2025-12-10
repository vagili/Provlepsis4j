# app/routers/gds_project_from_db.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Literal, Dict, Any

from ..db import run_data, run, current_database

router = APIRouter()

class ProjectFromDBBody(BaseModel):
    name: str                              
    nodeLabels: Optional[List[str]] = None 
    relationshipTypes: Optional[List[str]] = None 
    orientation: Literal["UNDIRECTED", "NATURAL", "REVERSE"] = "UNDIRECTED"
    dropIfExists: bool = True              

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
        run("CALL gds.graph.project($name, $labels, $rels)",
            {"name": body.name, "labels": labels, "rels": rel_cfg})
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Projection failed: {e}")

    return {"ok": True, "graphName": body.name}
