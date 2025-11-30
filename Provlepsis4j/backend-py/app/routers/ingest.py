from fastapi import APIRouter
from ..db import run

router = APIRouter()

@router.post("/sample")
def ingest_sample():
    # small toy graph of 6 users and a few edges
    run("CREATE CONSTRAINT IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE")
    run("""
    UNWIND $rows AS r
    MERGE (u:User {id:r.id})
    SET u += r.props
    """, {"rows":[
        {"id":0,"props":{"country":"GR","age":23}},
        {"id":1,"props":{"country":"GR","age":25}},
        {"id":2,"props":{"country":"DE","age":28}},
        {"id":3,"props":{"country":"FR","age":22}},
        {"id":4,"props":{"country":"DE","age":26}},
        {"id":5,"props":{"country":"GR","age":24}},
    ]})
    run("""
    UNWIND $rels AS r
    MATCH (a:User {id:r.src}), (b:User {id:r.dst})
    MERGE (a)-[:REL]->(b)
    """, {"rels":[
        {"src":0,"dst":1},{"src":1,"dst":2},{"src":2,"dst":3},
        {"src":3,"dst":4},{"src":4,"dst":5},{"src":0,"dst":5}
    ]})
    return {"ok": True, "nodes": 6, "rels": 6}
