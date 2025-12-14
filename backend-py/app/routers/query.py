# backend-py/app/routers/query.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
from neo4j.graph import Node as NeoNode, Relationship as NeoRel, Path as NeoPath
import re
from ..db import run_data, current_database

router = APIRouter()

USE_CLAUSE_PREFIX = False

class PresentBody(BaseModel):
    cypher: str
    params: Dict[str, Any] = {}

class FutureBody(BaseModel):
    cypher: str
    threshold: float = 0.0
    params: Dict[str, Any] = {}
    futureDb: Optional[str] = None

class BothBody(BaseModel):
    cypher: str
    params: Dict[str, Any] = {}
    futureDb: Optional[str] = None

class ModeBody(BothBody):
    mode: str = "table"  # "table" | "graph"

def _predicted_db_name(base: str) -> str:
    # Adjust if your predicted DB naming differs
    return f"{base}-predicted"

def _default_future_db() -> str:
    # default when the caller doesn't override
    return f"{current_database()}-predicted"

def _with_use_clause(dbname: str, cypher: str) -> str:
    q = cypher.lstrip()
    if q.lower().startswith("use "):
        return cypher
    return f"USE {dbname}\n{cypher}"

def _exec(cypher: str, params: Dict[str, Any], db: Optional[str] = None) -> List[Dict[str, Any]]:
    try:
        if db and USE_CLAUSE_PREFIX:
            cypher = _with_use_clause(db, cypher)
            rows = run_data(cypher, params)
        else:
            rows = run_data(cypher, params, db=db)
        return rows or []
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def _reject_gds_calls(cypher: str):
    if cypher.strip().lower().startswith(("call gds.", "yield gds.")):
        raise HTTPException(
            status_code=400,
            detail="This endpoint runs against databases only. Remove GDS calls (use pure Cypher)."
        )

from typing import Any, Dict, List, Tuple
from neo4j.graph import Node as NeoNode, Relationship as NeoRel, Path as NeoPath

import re

def _contains_shortest_path(cypher: str) -> bool:
    """
    Quick heuristic: does the cypher mention shortestPath(...)?
    Case-insensitive, simple match.
    """
    return bool(re.search(r"shortestpath\s*\(", cypher, flags=re.IGNORECASE))

def _contains_simple_path_return_p(cypher: str) -> bool:
    """
    Path queries like:

        MATCH p=(a)-[r]->(b)
        RETURN p
        RETURN p LIMIT 25
        MATCH p = shortestPath(...)
        RETURN p LIMIT 1

    i.e. a path bound to `p` and returned directly (optionally with LIMIT).
    """
    path_match = re.search(r"\bmatch\s+p\s*=", cypher, flags=re.IGNORECASE)

    ret_match = re.search(
        r"\breturn\s+(?:distinct\s+)?p(?:\s+limit\b|\s*$)",
        cypher,
        flags=re.IGNORECASE,
    )

    return bool(path_match and ret_match)

def _rewrite_shortest_path_cypher(cypher: str) -> str:
    """
    Take the user's query that contains:

        MATCH p = ...           -- plain path or shortestPath(...)
        RETURN p [SKIP ...] [LIMIT ...]

    and throw away their RETURN clause. Then we re-RETURN a clean
    {nodes, relationships} shape that is easy to convert into a graph.

    This assumes the path variable is named 'p'.
    """
    # Locate first RETURN
    m = re.search(r"\bRETURN\b", cypher, flags=re.IGNORECASE)

    limit_skip_clause = ""
    if m:
        # Part before RETURN
        prefix = cypher[:m.start()]

        # Everything after RETURN (e.g. " p LIMIT 25")
        suffix = cypher[m.end():]

        # Preserve (optional) SKIP / LIMIT in the same order
        clauses = []
        for kw in ("skip", "limit"):
            km = re.search(rf"\b{kw}\b\s+([^\s;]+)", suffix, flags=re.IGNORECASE)
            if km:
                expr = km.group(1).strip()
                if expr:
                    clauses.append(f"{kw.upper()} {expr}")

        if clauses:
            limit_skip_clause = "\n" + " ".join(clauses)
    else:
        prefix = cypher

    # Ensure trailing semicolon doesn't break anything
    prefix = prefix.rstrip().rstrip(";")

    # Rebuild query to expose nodes/relationships with elementIds
    rewritten = f"""
{prefix}
WITH p
WITH p, nodes(p) AS ns, relationships(p) AS rs
RETURN
  [n IN ns | {{
    elementId: elementId(n),
    labels: labels(n),
    properties: properties(n)
  }}] AS nodes,
  [r IN rs | {{
    elementId: elementId(r),
    type: type(r),
    startNodeElementId: elementId(startNode(r)),
    endNodeElementId: elementId(endNode(r)),
    properties: properties(r)
  }}] AS relationships{limit_skip_clause}
"""
    return rewritten.strip()


def _shortest_path_graph(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Consume rows of the form:
      { nodes: [ {...}, ... ], relationships: [ {...}, ... ] }
    and turn into our standard {nodes:[], edges:[]} structure.
    """
    nodes_by_id: Dict[str, Dict[str, Any]] = {}
    edges_by_id: Dict[str, Dict[str, Any]] = {}

    def add_node(n: Dict[str, Any]):
        nid = str(
            n.get("elementId")
            or n.get("element_id")
            or n.get("id", "")
        )
        if not nid:
            return

        labels = n.get("labels") or []
        props = n.get("properties") or {}

        # business id from Neo4j property `id`
        node_id_prop = None
        if "id" in props:
            node_id_prop = props["id"]

        if nid not in nodes_by_id:
            node: Dict[str, Any] = {"id": nid, "labels": labels}

            # expose the business id under a dedicated field
            if node_id_prop is not None:
                node["nodeId"] = node_id_prop

            # copy remaining properties
            for k, v in props.items():
                if k not in ("id", "labels", "nodeId"):
                    node[k] = v

            nodes_by_id[nid] = node

    def add_rel(r: Dict[str, Any]):
        rid = str(
            r.get("elementId")
            or r.get("element_id")
            or r.get("id", "")
        )
        sid = str(
            r.get("startNodeElementId")
            or r.get("start")
            or r.get("startNode")
            or ""
        )
        tid = str(
            r.get("endNodeElementId")
            or r.get("end")
            or r.get("endNode")
            or ""
        )
        if not sid or not tid:
            return

        rtype = r.get("type") or ""
        if not rid:
            rid = f"{sid}->{tid}:{rtype}"

        props = r.get("properties") or {}
        add_node({"elementId": sid, "labels": [], "properties": {}})
        add_node({"elementId": tid, "labels": [], "properties": {}})

        edge = {
            "id": rid,
            "type": rtype,
            "source": sid,
            "target": tid,
        }
        for k, v in props.items():
            if k not in ("id", "labels"):
                edge[k] = v
        edges_by_id[rid] = edge

    for row in rows or []:
        if not isinstance(row, dict):
            continue
        ns = row.get("nodes") or []
        rs = row.get("relationships") or []
        if isinstance(ns, list):
            for n in ns:
                if isinstance(n, dict):
                    add_node(n)
        if isinstance(rs, list):
            for r in rs:
                if isinstance(r, dict):
                    add_rel(r)

        # also handle a wrapped shape like {path: {nodes: [...], relationships: [...]}}
        path = row.get("path")
        if isinstance(path, dict):
            for n in path.get("nodes", []) or []:
                if isinstance(n, dict):
                    add_node(n)
            for r in path.get("relationships", []) or []:
                if isinstance(r, dict):
                    add_rel(r)

    return {
        "nodes": list(nodes_by_id.values()),
        "edges": list(edges_by_id.values()),
    }

def _graphify(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    nodes_by_id: Dict[str, Dict[str, Any]] = {}
    edges_by_id: Dict[str, Dict[str, Any]] = {}

    def node_id_from(x: Any) -> str:
        # Neo4j python objects
        if isinstance(x, NeoNode):
            return str(getattr(x, "element_id", getattr(x, "id", "")))
        # Dict-like
        if isinstance(x, dict):
            # common fields we might see
            for k in ("elementId", "element_id", "identity", "id"):
                if k in x and x[k] is not None:
                    return str(x[k])
        return ""


    def add_node(x: Any):
        # Neo4j python node
        if isinstance(x, NeoNode):
            nid = str(getattr(x, "element_id", getattr(x, "id", "")))
            labels = list(getattr(x, "labels", []))
            props = dict(getattr(x, "_properties", {}))
        elif isinstance(x, dict):
            nid = node_id_from(x)
            labels = x.get("labels") if isinstance(x.get("labels"), list) else []
            # two shapes:
            #  - flattened props (many keys at top level)
            #  - wrapped under {"properties": {...}}
            if "properties" in x and isinstance(x["properties"], dict):
                props = dict(x["properties"])
            else:
                props = {
                    k: v
                    for k, v in x.items()
                    if k
                    not in (
                        "labels",
                        "type",
                        "start",
                        "end",
                        "startNode",
                        "endNode",
                        "startNodeElementId",
                        "endNodeElementId",
                        "identity",
                        "elementId",
                        "element_id",
                        "id",
                        "segments",
                        "nodes",
                        "relationships",
                    )
                }
        else:
            nid = ""
            labels, props = [], {}

        if not nid:
            return

        # business id property
        node_id_prop = None
        if isinstance(x, NeoNode):
            raw_props = getattr(x, "_properties", {}) or {}
            if "id" in raw_props:
                node_id_prop = raw_props["id"]
        elif isinstance(x, dict):
            if "properties" in x and isinstance(x["properties"], dict):
                if "id" in x["properties"]:
                    node_id_prop = x["properties"]["id"]
            elif "id" in x:
                node_id_prop = x["id"]

        if nid not in nodes_by_id:
            node_obj: Dict[str, Any] = {"id": nid, "labels": labels}

            # expose the business id under a safe field
            if node_id_prop is not None:
                node_obj["nodeId"] = node_id_prop

            # merge other props
            for k, v in props.items():
                if k not in ("id", "labels", "nodeId"):
                    node_obj[k] = v

            nodes_by_id[nid] = node_obj


    def add_rel(r: Any, src: Any, trg: Any):
        sid = node_id_from(src)
        tid = node_id_from(trg)
        if not sid or not tid:
            return

        # Neo4j python relationship
        if isinstance(r, NeoRel):
            rtype = getattr(r, "type", "")
            rid = str(getattr(r, "element_id", getattr(r, "id", f"{sid}->{tid}:{rtype}")))
            props = dict(getattr(r, "_properties", {}))
        elif isinstance(r, dict):
            rtype = r.get("type", "") or ""
            rid = str(r.get("elementId") or r.get("element_id") or r.get("identity") or r.get("id") or f"{sid}->{tid}:{rtype}")
            # edge dict might already carry start/end ids
            if not sid:
                sid = str(r.get("startNodeElementId") or r.get("start") or r.get("startNode") or "")
            if not tid:
                tid = str(r.get("endNodeElementId") or r.get("end") or r.get("endNode") or "")
            if not sid or not tid:
                return
            props = dict(r.get("properties", {})) if isinstance(r.get("properties"), dict) else {
                # also accept flattened r-props
                k: v for k, v in r.items() if k not in (
                    "labels", "type", "start", "end", "startNode", "endNode",
                    "startNodeElementId", "endNodeElementId", "identity",
                    "elementId", "element_id", "id", "segments", "nodes", "relationships"
                )
            }
        else:
            rtype = str(r)
            rid = f"{sid}->{tid}:{rtype}"
            props = {}

        add_node(src)
        add_node(trg)
        edges_by_id[rid] = {"id": rid, "type": rtype, "source": sid, "target": tid, **props}

    def consume_value(v: Any):
        # 1) Neo4j python objects
        if isinstance(v, NeoNode):
            add_node(v); return True
        if isinstance(v, NeoRel):
            return True
        if isinstance(v, NeoPath):
            for i in range(len(v.relationships)):
                a = v.nodes[i]; b = v.nodes[i+1]; r = v.relationships[i]
                add_rel(r, a, b)
            return True

        # 2) List-like
        if isinstance(v, list):
            if len(v) == 3 and isinstance(v[0], dict) and isinstance(v[2], dict) and isinstance(v[1], (str,)):
                add_rel(v[1], v[0], v[2])
                return True
            for x in v:
                consume_value(x)
            return True

        # 3) Dict-like
        if isinstance(v, dict):
            if all(k in v for k in ("a", "b", "r")):
                add_rel(v["r"], v["a"], v["b"])
                return True

            # Node-like dicts
            if node_id_from(v):
                add_node(v)
                if "type" in v and ("startNodeElementId" in v or "start" in v or "startNode" in v) and ("endNodeElementId" in v or "end" in v or "endNode" in v):
                    add_rel(v, {"id": v.get("startNodeElementId") or v.get("start") or v.get("startNode")},
                               {"id": v.get("endNodeElementId") or v.get("end") or v.get("endNode")})
                return True

            # Relationship-like dicts with explicit endpoints
            if v.get("type") and (v.get("start") or v.get("startNode") or v.get("startNodeElementId")) and (v.get("end") or v.get("endNode") or v.get("endNodeElementId")):
                src = {"id": v.get("startNodeElementId") or v.get("start") or v.get("startNode")}
                dst = {"id": v.get("endNodeElementId") or v.get("end") or v.get("endNode")}
                add_rel(v, src, dst)
                return True

            # Path-like dicts: segments
            segs = v.get("segments")
            if isinstance(segs, list) and segs:
                for s in segs:
                    a = s.get("start"); b = s.get("end"); r = s.get("relationship")
                    if a and b and r:
                        add_rel(r, a, b)
                return True

            # Fallback
            for x in v.values():
                consume_value(x)
            return True

        return False

    for rec in records or []:
        for val in rec.values():
            consume_value(val)

    return {"nodes": list(nodes_by_id.values()), "edges": list(edges_by_id.values())}


@router.post("/present")
def query_present(body: PresentBody):
    base_db = current_database()
    _reject_gds_calls(body.cypher)
    params = dict(body.params)
    return {"records": _exec(body.cypher, params, db=base_db)}

@router.post("/future")
def query_future(body: FutureBody):
    _reject_gds_calls(body.cypher)
    params = dict(body.params)
    params.setdefault("thr", float(body.threshold))
    # allow override, else default
    predicted_db = body.futureDb or _default_future_db()
    rows = _exec(body.cypher, params, db=predicted_db)
    return {"records": rows, "db": predicted_db}

@router.post("/both")
def query_both(body: BothBody):
    _reject_gds_calls(body.cypher)
    params = dict(body.params)
    base_db = current_database()
    predicted_db = body.futureDb or _default_future_db()

    present_rows = _exec(body.cypher, params, db=base_db)
    future_rows  = _exec(body.cypher, params, db=predicted_db)

    # Keep backward compat: include both "future" and "predicted"
    return {
        "present": present_rows,
        "future": future_rows,             
        "predicted": future_rows, 
        "databases": {"present": base_db, "future": predicted_db, "predicted": predicted_db},
    }

# ---- graph mode variant ----
class ModeBody(BothBody):
    mode: str = "table"  # "table" | "graph"

@router.post("/both2")
def query_both2(body: ModeBody):
    _reject_gds_calls(body.cypher)
    params = dict(body.params)
    base_db = current_database()
    predicted_db = body.futureDb or _default_future_db()

    # --- Non-graph mode: behave exactly as before ---
    if body.mode != "graph":
        present_rows = _exec(body.cypher, params, db=base_db)
        future_rows  = _exec(body.cypher, params, db=predicted_db)
        return {
            "present": present_rows,
            "future": future_rows,               
            "predicted": future_rows,                 
            "databases": {
                "present": base_db,
                "future": predicted_db,
                "predicted": predicted_db,
            },
            "ok": True,
        }

    # --- Graph mode ---

    # 1) Path queries that return a single path variable `p`
    #    - either shortestPath(...) or a plain MATCH p=(...)-[r]->(...)
    if _contains_shortest_path(body.cypher) or _contains_simple_path_return_p(body.cypher):
        # Rewrite to a shape that exposes {nodes, relationships}
        path_cypher = _rewrite_shortest_path_cypher(body.cypher)

        present_rows = _exec(path_cypher, params, db=base_db)
        future_rows  = _exec(path_cypher, params, db=predicted_db)

        present_graph = _shortest_path_graph(present_rows)
        future_graph  = _shortest_path_graph(future_rows)

        return {
            "present": present_graph,
            "future": future_graph,
            "predicted": future_graph,
            "databases": {
                "present": base_db,
                "future": predicted_db,
                "predicted": predicted_db,
            },
            "ok": True,
        }


    # 2) All other queries:
    present_rows = _exec(body.cypher, params, db=base_db)
    future_rows  = _exec(body.cypher, params, db=predicted_db)

    return {
        "present": _graphify(present_rows),
        "future": _graphify(future_rows),      
        "predicted": _graphify(future_rows),   
        "databases": {
            "present": base_db,
            "future": predicted_db,
            "predicted": predicted_db,
        },
        "ok": True,
    }

@router.get("/dbs")
def query_dbs():
    base = current_database()
    return {
        "present": base,
        "predicted": f"{base}-predicted",
        "future": f"{base}-predicted",  
    }
