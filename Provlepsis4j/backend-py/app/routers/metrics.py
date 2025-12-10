# app/logic/metrics.py
from fastapi import APIRouter
from pydantic import BaseModel
from ..db import run

router = APIRouter()

class MetricsBody(BaseModel):
    threshold: float = 0.5

@router.post("/lp")
def metrics_lp(body: MetricsBody):
    tp = run("""
        MATCH (a)-[t:REL_TEST]->(b)
        MATCH (a)-[p:REL_PRED]->(b)
        WHERE p.score >= $th
        RETURN count(*) AS c
    """, {"th": body.threshold}).single()["c"]

    fp = run("""
        MATCH (a)-[p:REL_PRED]->(b)
        WHERE p.score >= $th AND NOT (a)-[:REL_TEST]->(b) AND NOT (a)-[:REL_TRAIN]->(b)
        RETURN count(*) AS c
    """, {"th": body.threshold}).single()["c"]

    fn = run("""
        MATCH (a)-[t:REL_TEST]->(b)
        WHERE NOT EXISTS {
          MATCH (a)-[p:REL_PRED]->(b) WHERE p.score >= $th
        }
        RETURN count(*) AS c
    """, {"th": body.threshold}).single()["c"]

    tp = int(tp); fp = int(fp); fn = int(fn)
    precision = 0 if (tp+fp)==0 else tp/(tp+fp)
    recall = 0 if (tp+fn)==0 else tp/(tp+fn)
    f1 = 0 if (precision+recall)==0 else 2*precision*recall/(precision+recall)

    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1, "threshold": body.threshold}

class TopPredBody(BaseModel):
    nodeId: int
    k: int = 50

@router.post("/topPredicted")
def top_predicted(body: TopPredBody):
    res = run("""
        MATCH (n)-[p:REL_PRED]->(m)
        WHERE id(n)=$id
        RETURN m AS node, p.score AS score
        ORDER BY score DESC LIMIT $k
    """, {"id": body.nodeId, "k": body.k})

    rows = []
    for r in res:
        rows.append({"node": r["node"], "score": r["score"]})
    return {"rows": rows}
