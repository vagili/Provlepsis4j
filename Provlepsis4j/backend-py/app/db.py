# app/db.py
import os
from typing import Any, Dict, Optional, Iterable
from neo4j import GraphDatabase, Result, Driver
from dotenv import load_dotenv
from neo4j.graph import Node as NeoNode, Relationship as NeoRel, Path as NeoPath
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
DEFAULT_DB = os.getenv("NEO4J_DATABASE", "neo4j")

# Global driver (can be set at runtime via /config/neo4j)
driver: Optional[Driver] = None

# -------- current database selection (runtime) --------
_current_db = DEFAULT_DB

def current_database() -> str:
    return _current_db

def set_database(name: str) -> None:
    global _current_db
    _current_db = name

def set_driver_config(uri: str, user: str, password: str, database: Optional[str] = None) -> None:
    """Called by /config/neo4j to set or change the connection at runtime."""
    global driver, _current_db
    if driver is not None:
        driver.close()
    driver = GraphDatabase.driver(uri, auth=(user, password))
    if database:
        _current_db = database

# If env vars exist, eagerly configure (dev convenience). Otherwise, wait for /config/neo4j.
if NEO4J_URI and NEO4J_USER and NEO4J_PASSWORD:
    set_driver_config(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, DEFAULT_DB)

def _require_driver():
    if driver is None:
        raise RuntimeError("Neo4j connection is not configured. POST /config/neo4j first.")

# -------- admin: list databases (must run against 'system') --------
def list_databases():
    _require_driver()
    with driver.session(database="system") as s:
        q = """
        SHOW DATABASES
        YIELD name, currentStatus, access, role, default, home, address
        RETURN name, currentStatus, access, role, default, home, address
        ORDER BY name
        """
        return s.run(q).data()

# -------- query helpers --------
def run(cypher: str, params: Optional[Dict[str, Any]] = None, db: Optional[str] = None) -> Result:
    _require_driver()
    with driver.session(database=db or _current_db) as session:
        return session.run(cypher, params or {})

def run_data(query: str, params: Optional[Dict[str, Any]] = None,
             db: Optional[str] = None) -> list[Dict[str, Any]]:
    _require_driver()
    with driver.session(database=db or _current_db) as session:
        result = session.run(query, params or {})
        rows = []
        for record in result:
            row = {}
            for key, value in record.items():
                row[key] = _encode_neo4j_value(value)
            rows.append(row)
        return rows

def run_value(query: str, params: Optional[Dict[str, Any]] = None, db: Optional[str] = None, default: Any = None) -> Any:
    _require_driver()
    with driver.session(database=db or _current_db) as session:
        result = session.run(query, params or {})
        try:
            return result.single(strict=False).value()
        except Exception:
            return default
        
def _encode_neo4j_value(v):
    # primitives are fine
    if v is None or isinstance(v, (bool, int, float, str)):
        return v

    # lists
    if isinstance(v, (list, tuple)):
        return [_encode_neo4j_value(x) for x in v]

    # dicts (params, etc.)
    if isinstance(v, dict):
        return {k: _encode_neo4j_value(val) for k, val in v.items()}

    # Node
    if isinstance(v, NeoNode):
        return {
            "elementId": v.element_id,
            "labels": list(v.labels),
            "properties": dict(getattr(v, "_properties", {})),
        }

    # Relationship
    if isinstance(v, NeoRel):
        props = dict(getattr(v, "_properties", {}))
        # start_node / end_node are available on NeoRel
        start_id = getattr(getattr(v, "start_node", None), "element_id", None)
        end_id   = getattr(getattr(v, "end_node", None), "element_id", None)
        return {
            "elementId": v.element_id,
            "type": v.type,
            "startNodeElementId": start_id,
            "endNodeElementId": end_id,
            "properties": props,
        }

    # Path
    if isinstance(v, NeoPath):
        return {
            "nodes": [_encode_neo4j_value(n) for n in v.nodes],
            "relationships": [_encode_neo4j_value(r) for r in v.relationships],
        }

    # fallback: string representation
    return str(v)
