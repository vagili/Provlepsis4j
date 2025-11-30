from fastapi import APIRouter
from pydantic import BaseModel
from ..db import run
from .gds_context import resolve_graph_name

router = APIRouter()

class Node2VecBody(BaseModel):
    graphName: str | None = None  # optional – falls back to selected graph
    embeddingDimension: int = 128
    walkLength: int = 80
    walksPerNode: int = 10
    returnFactor: float = 1.0      # was p
    inOutFactor: float = 1.0       # was q
    writeProperty: str = "embedding_n2v_128"

@router.post("/write")
def node2vec_write(body: Node2VecBody):
    g = "trainGraph"

    cfg = {
        "embeddingDimension": body.embeddingDimension,
        "walkLength": body.walkLength,
        "walksPerNode": body.walksPerNode,
        "returnFactor": body.returnFactor,
        "inOutFactor": body.inOutFactor,
        "writeProperty": body.writeProperty,
    }

    # Minimal call: no YIELD, no RETURN – just do the write.
    stmt = "CALL gds.node2vec.write($g, $cfg)"
    run(stmt, {"g": g, "cfg": cfg})

    return {"ok": True, "graphName": g, "writeProperty": body.writeProperty}
