# app/routers/config.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from neo4j import GraphDatabase
from .. import db

router = APIRouter()

class Neo4jConfig(BaseModel):
    uri: str
    user: str
    password: str
    database: str = "neo4j"

@router.post("/neo4j")
def set_neo4j(cfg: Neo4jConfig):
    try:
        probe = GraphDatabase.driver(cfg.uri, auth=(cfg.user, cfg.password))
        with probe.session(database=cfg.database) as s:
            s.run("RETURN 1").consume()
        probe.close()
    except Exception as e:
        raise HTTPException(400, detail=f"Connection failed: {e}")

    db.set_driver_config(cfg.uri, cfg.user, cfg.password, cfg.database)
    db.set_database(cfg.database)
    return {"ok": True, "database": cfg.database}

@router.get("/neo4j")
def get_status():
    try:
        name = db.current_database()
        _ = db.run_value("RETURN 1", default=None)
        return {"configured": True, "database": name}
    except Exception:
        return {"configured": False}
