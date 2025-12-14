# app/logic/gds.py
from fastapi import APIRouter
from pydantic import BaseModel
from ..db import run

router = APIRouter()

class ProjectBody(BaseModel):
    name: str = "g"
    labels: list[str] | None = None
    rels: dict | None = None
    nodeProperties: list[str] | None = None

@router.post("/graph/project")
def project_graph(body: ProjectBody):
    labels = body.labels or ["*"]
    rels = body.rels or {"REL_TRAIN": {"type": "REL_TRAIN", "orientation": "UNDIRECTED"}}
    node_props = body.nodeProperties or []

    try:
        run("CALL gds.graph.drop($g, false)", {"g": body.name})
    except Exception:
        pass

    cypher = "CALL gds.graph.project($g, $labels, $rels, {nodeProperties:$props})"
    run(cypher, {"g": body.name, "labels": labels, "rels": rels, "props": node_props})
    return {"graphName": body.name}
