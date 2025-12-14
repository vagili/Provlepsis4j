# app/logic/gds_meta.py
from fastapi import APIRouter, Depends, HTTPException
from typing import List, Dict, Any
from ..db import run_data

router = APIRouter()

@router.get("/graphs")
def list_projected_graphs() -> List[Dict[str, Any]]:
    q = """
    CALL gds.graph.list()
    YIELD graphName, nodeCount, relationshipCount, creationTime, schema, memoryUsage
    RETURN graphName, nodeCount, relationshipCount, creationTime, schema, memoryUsage
    ORDER BY graphName ASC
    """
    try:
        recs = run_data(q)
    except Exception as e:
        raise HTTPException(status_code=400, detail={
            'message': f'Server failed to list projected graphs: {e}',
            'HINT': 'Ensure the Desktop DBMS has the Graph Data Science plugin enabled and is the database configured in NEO4J_DATABASE. In Browser try: CALL gds.version();'
        })
    out = []
    for r in recs:
        out.append({
            "graphName": r.get("graphName"),
            "nodeCount": r.get("nodeCount"),
            "relationshipCount": r.get("relationshipCount"),
            "creationTime": str(r.get("creationTime")),
            "memoryUsage": r.get("memoryUsage"),
            "schema": r.get("schema"),
        })
    return out
