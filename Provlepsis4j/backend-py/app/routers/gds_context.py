from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

_graph_context = {"graphName": None}

class GraphCtxIn(BaseModel):
    graphName: str

@router.get("/graph-context")
def get_graph_context():
    return {"graphName": _graph_context["graphName"]}

@router.post("/graph-context")
def set_graph_context(body: GraphCtxIn):
    name = (body.graphName or "").strip()
    if not name:
        raise HTTPException(400, "graphName is required")
    _graph_context["graphName"] = name
    return {"graphName": name}

# utility for other routers to resolve name
def resolve_graph_name(graphName: str | None) -> str:
    name = (graphName or _graph_context.get("graphName") or "").strip()
    if not name:
        raise HTTPException(400, "graphName is required (either in body or set via /gds/graph-context)")
    return name
