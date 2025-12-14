# app/routers/split.py
from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, Tuple, List
from uuid import uuid4
import re
import random
from math import inf

from ..db import run, run_data
from neo4j.exceptions import Neo4jError

router = APIRouter()

# =========================
# Feature helpers (optional)
# =========================
try:
    # Should return List[str] of node property keys to project
    from .feature import get_projection_props  
    _FEATURES_OK = True
except Exception:
    get_projection_props = None  
    _FEATURES_OK = False

# Per-DB feature cache (+ light meta so we can auto-invalidate)
_FEATURE_PROPS_CACHE: Dict[str, List[str]] = {}
_FEATURE_PROPS_META: Dict[str, Dict[str, Any]] = {}

def _current_db_name() -> str:
    try:
        row = run_data("CALL db.info() YIELD name RETURN name")[0]
        return row["name"] or "neo4j"
    except Exception:
        # Fallback: default DB name
        return "neo4j"

def clear_feature_cache(db: Optional[str] = None):
    """
    Clear cached feature projection list.
    If db is None, clear all.
    """
    if db:
        _FEATURE_PROPS_CACHE.pop(db, None)
        _FEATURE_PROPS_META.pop(db, None)
    else:
        _FEATURE_PROPS_CACHE.clear()
        _FEATURE_PROPS_META.clear()

# ---- property discovery & filtering ----

def _all_node_labels() -> List[str]:
    rows = run_data("CALL db.labels() YIELD label RETURN collect(label) AS labels")
    return rows[0]["labels"] if rows else []

def _autodiscover_node_props(labels: List[str]) -> List[str]:
    """
    Heuristic discovery: union of keys() across a sample of nodes for the given labels.
    """
    if not labels:
        return []
    label_filter = " OR ".join([f"n:`{lbl}`" for lbl in labels])
    rows = run_data(f"""
        MATCH (n)
        WHERE {label_filter}
        WITH n LIMIT 2000
        WITH apoc.coll.toSet(apoc.coll.flatten(collect(keys(n)))) AS ks
        UNWIND ks AS k
        WITH DISTINCT k
        WHERE NOT k IN ['_id','_tmp','_ts']   // ignore obviously internal-ish keys if present
        RETURN collect(k) AS props
    """)
    return rows[0].get("props", []) if rows else []

def ensure_feature_props_loaded(force: bool = False) -> List[str]:
    """
    1) Try feature.get_projection_props()
    2) If empty/unavailable, auto-discover from the data
    Cache is per-DB and auto-invalidates if label set changes.
    """
    db = _current_db_name()

    # Quick label snapshot to detect schema changes
    labels = _all_node_labels()
    label_sig = tuple(sorted(labels))

    if not force:
        cached = _FEATURE_PROPS_CACHE.get(db)
        meta = _FEATURE_PROPS_META.get(db)
        if cached is not None and meta and meta.get("labels") == label_sig:
            return cached

    base: List[str] = []

    if _FEATURES_OK and get_projection_props:
        try:
            base = list(sorted(set(get_projection_props() or [])))
        except Exception:
            base = []

    if not base:
        base = _autodiscover_node_props(labels)

    # Save (or overwrite) cache for this DB
    _FEATURE_PROPS_CACHE[db] = base
    _FEATURE_PROPS_META[db] = {"labels": label_sig}
    return base


_ALLOWED_SCALARS = (int, float, bool, str)

def _filter_supported_node_props(props: List[str]) -> List[str]:
    """
    Keep props whose sampled non-null values look GDS-acceptable.
    If filtering would drop everything, fall back to the original list so we still project.
    """
    if not props:
        return []

    safe: List[str] = []
    for p in props:
        rows = run_data(
            "WITH $k AS k MATCH (n) WHERE n[k] IS NOT NULL RETURN n[k] AS v LIMIT 500",
            {"k": p},
        )

        # If we saw no non-null samples, keep it (features can be sparse)
        if not rows:
            safe.append(p)
            continue

        ok = True
        for r in rows:
            v = r.get("v")
            if v is None:
                continue
            if isinstance(v, _ALLOWED_SCALARS):
                continue
            if isinstance(v, list):
                # allow numeric arrays (embeddings, etc.)
                if all((x is None) or isinstance(x, (int, float)) for x in v):
                    continue
            ok = False
            break

        if ok:
            safe.append(p)

    return safe if safe else list(props)

# ----- Embedding property exclusions -----
# Exact property names we should NOT project as node features
_EMBED_PROP_NAMES = {
    "FastRP", "Node2Vec", "GraphSAGE", "HashGNN",
    # common variants / lowercase
    "fastrp", "node2vec", "graphsage", "hashgnn",
    "embedding_n2v_128", "embedding_fastrp_256", "embedding_sage_64", "embedding_hash_128",
}

# Conservative prefixes to exclude
_EMBED_PROP_PREFIXES = {
    "fastrp_", "node2vec_", "graphsage_", "hashgnn_",
    "emb_", "embedding_",
}

def _drop_embedding_props(props: List[str]) -> List[str]:
    """Remove properties that look like embedding outputs."""
    cleaned = []
    for p in props:
        if p in _EMBED_PROP_NAMES:
            continue
        if any(p.startswith(pref) for pref in _EMBED_PROP_PREFIXES):
            continue
        cleaned.append(p)
    return cleaned

# =========================
# GDS projection utilities
# =========================
def _drop_in_memory_gds_graph(name: str):
    try:
        run("CALL gds.graph.drop($name, false)", {"name": name})
    except Exception:
        pass

def _project_with_native(
    graph_name: str,
    rel_types: List[str],
    node_props: List[str]
) -> Dict[str, Any]:
    """
    Project using gds.graph.project with per-label nodeProjection
    so node properties are actually included.
    NOTE: This projects ALL labels; we prefer the temp-endpoint-label variant below
    to include only participating nodes.
    """
    labels = _all_node_labels()
    if not labels:
        return {"graphName": graph_name, "nodeCount": 0, "relationshipCount": 0, "note": "no labels in DB"}

    # Per-label nodeProjection including properties
    node_projection = {lbl: ({"properties": node_props} if node_props else {}) for lbl in labels}

    _drop_in_memory_gds_graph(graph_name)

    rows = run_data(
        """
        CALL gds.graph.project(
          $name,
          $nodeProjection,
          $relationshipTypes
        )
        YIELD graphName, nodeCount, relationshipCount
        RETURN graphName, nodeCount, relationshipCount
        """,
        {
            "name": graph_name,
            "nodeProjection": node_projection,
            "relationshipTypes": rel_types,
        },
    )
    return rows[0] if rows else {"graphName": graph_name, "nodeCount": 0, "relationshipCount": 0}

# ---------- temp-endpoint label projection (keeps only participating nodes) ----------
def _safe_label(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", s)

def _tag_split_endpoints(temp_label: str, reltype: str):
    temp_label = _safe_label(temp_label)
    run(f"""
      MATCH ()-[r:`{reltype}`]-()
      WITH startNode(r) AS a, endNode(r) AS b
      SET a:`{temp_label}`, b:`{temp_label}`
    """)

def _untag_split_endpoints(temp_label: str):
    temp_label = _safe_label(temp_label)
    run(f"MATCH (n:`{temp_label}`) REMOVE n:`{temp_label}`")

def _project_with_label_only(
    graph_name: str,
    endpoint_label: str,
    rel_types: List[str],
    node_props: List[str],
) -> Dict[str, Any]:
    """
    Project only nodes having `endpoint_label`, and only the provided rel_types.
    Node properties are included as in your native projection.
    """
    endpoint_label = _safe_label(endpoint_label)
    node_projection = {endpoint_label: ({"properties": node_props} if node_props else {})}
    _drop_in_memory_gds_graph(graph_name)
    rows = run_data(
        """
        CALL gds.graph.project(
          $name,
          $nodeProjection,
          $relationshipTypes
        )
        YIELD graphName, nodeCount, relationshipCount
        RETURN graphName, nodeCount, relationshipCount
        """,
        {
            "name": graph_name,
            "nodeProjection": node_projection,
            "relationshipTypes": rel_types,
        },
    )
    return rows[0] if rows else {"graphName": graph_name, "nodeCount": 0, "relationshipCount": 0}

# =========================
# Connected-train helpers (base graph UNDIRECTED)
# =========================

def _relationship_types_original() -> List[str]:
    rows = run_data("""
      CALL db.relationshipTypes() YIELD relationshipType
      RETURN relationshipType AS type
    """)
    return [r["type"] for r in rows if not str(r["type"]).startswith("SPLIT_")]

def _project_base_for_split(gname: str) -> Optional[str]:
    """
    Project the ORIGINAL graph (no SPLIT_ rels) as UNDIRECTED topology,
    so BFS/WCC see weak connectivity.
    """
    labels = _all_node_labels()
    if not labels:
        return None
    rel_types = _relationship_types_original()
    if not rel_types:
        return None

    # UNDIRECTED relationship projection to ensure BFS covers weak connectivity
    rel_proj = { rt: {"orientation": "UNDIRECTED"} for rt in rel_types }

    _drop_in_memory_gds_graph(gname)
    run_data("""
      CALL gds.graph.project($g, $nodeProj, $relProj)
      YIELD graphName
      RETURN graphName
    """, {
      "g": gname,
      "nodeProj": {lbl: {} for lbl in labels},  # topology only
      "relProj": rel_proj,
    })
    return gname

def _backbone_edges_with_bfs(gname: str) -> List[Tuple[int,int]]:
    """
    Return undirected node-id pairs (u,v) (u<v) that form a spanning forest
    across all WCCs in the base projection.
    """
    try:
        seeds = run_data("""
          CALL gds.wcc.stream($g)
          YIELD nodeId, componentId
          WITH componentId, collect(nodeId) AS ns
          RETURN ns[0] AS seed
        """, {"g": gname})
        backbone: List[Tuple[int,int]] = []
        for row in seeds:
            seed = int(row["seed"])
            rows = run_data("""
              CALL gds.beta.bfs.stream($g, {sourceNode: $seed})
              YIELD nodeId, parentId
              WHERE parentId <> -1
              WITH gds.util.asNode(nodeId)  AS n,
                   gds.util.asNode(parentId) AS p
              RETURN id(p) AS u, id(n) AS v
            """, {"g": gname, "seed": seed})
            for r in rows:
                u = int(r["u"]); v = int(r["v"])
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                backbone.append((u, v))
        return list({e for e in backbone})
    except Neo4jError:
        # Optional: APOC fallback could go here if desired.
        return []

def _create_train_edges_from_node_ids(
    backbone_pairs: List[Tuple[int, int]],
    reltype: str,
    batch_size: int = 5000
) -> int:
    """
    Create mirrored TRAIN edges for the given undirected backbone pairs (u,v) of Neo4j node IDs.
    - Skips any pair that already has a SPLIT_* edge between the endpoints (no overlaps).
    - Uses MERGE to be idempotent.
    Returns the number of DISTINCT undirected pairs written for `reltype`.
    """
    if not backbone_pairs:
        return 0

    # Write in batches; guard against overlaps with any existing SPLIT_* between same endpoints
    for i in range(0, len(backbone_pairs), batch_size):
        chunk = backbone_pairs[i:i + batch_size]
        run(f"""
          UNWIND $pairs AS p
          MATCH (a) WHERE id(a) = p[0]
          MATCH (b) WHERE id(b) = p[1]
          WITH a, b
          WHERE NOT EXISTS {{ MATCH (a)-[r]-(b) WHERE type(r) STARTS WITH 'SPLIT_' }}
          MERGE (a)-[:`{reltype}`]->(b)
          MERGE (b)-[:`{reltype}`]->(a)
        """, {"pairs": [[u, v] for (u, v) in chunk]})

    # Return distinct undirected pair count for this reltype
    rows = run_data(f"""
      MATCH ()-[r:`{reltype}`]-()
      WITH id(startNode(r)) AS s, id(endNode(r)) AS t
      WITH CASE WHEN s < t THEN [s, t] ELSE [t, s] END AS p
      RETURN count(DISTINCT p) AS k
    """)
    return int(rows[0]["k"]) if rows else 0


# =========================
# Split materialization
# =========================
def _mk_run_suffix() -> str:
    # Relationship type suffix must be ascii letters/digits/underscore only.
    return uuid4().hex[:6].upper()

class _SplitTypes(Dict[str, str]):
    test: str
    val: str
    train: str

def _build_run_types(run_suffix: str) -> _SplitTypes:
    return {
        "test":  f"SPLIT_TEST_{run_suffix}",
        "val":   f"SPLIT_VAL_{run_suffix}",
        "train": f"SPLIT_TRAIN_{run_suffix}",
    }

def _materialize_temp_split_relationships(
    test_holdout: float,
    val_holdout: Optional[float],
    types: _SplitTypes,
    batch_size: int = 5000
) -> Dict[str, int]:
    """
    Random split (fallback). Kept for completeness behind ensureConnected flag.
    """
    t = max(0.0, min(1.0, float(test_holdout)))
    v = max(0.0, min(1.0, float(val_holdout or 0.0)))

    # Count P over UNDIRECTED pairs, ignoring any SPLIT_* rels
    row = run_data("""
        MATCH (a)-[r]-(b)
        WHERE NOT type(r) STARTS WITH 'SPLIT_'
        WITH
          CASE WHEN id(a) < id(b) THEN id(a) ELSE id(b) END AS s,
          CASE WHEN id(a) < id(b) THEN id(b) ELSE id(a) END AS t,
          type(r) AS typ
        RETURN count(DISTINCT [s,t,typ]) AS P
    """)[0]
    P = int(row["P"]) if row and row.get("P") is not None else 0

    from math import floor
    k_test = int(floor(P * t))
    k_val  = int(floor(P * v))
    k_val  = max(0, min(P - k_test, k_val))
    k_train = max(0, P - k_test - k_val)

    def _create_pairs(limit_k: int, reltype: str, exclude_types: List[str], batch_size: int = 5000) -> int:
        if limit_k <= 0:
            return 0

        cypher = f"""
        CALL {{
          WITH $k AS k, $ex AS ex
          MATCH (a)-[r]-(b)
          WHERE NOT type(r) STARTS WITH 'SPLIT_'
            AND ALL(x IN ex WHERE NOT EXISTS {{
              MATCH (a)-[r2]-(b) WHERE type(r2) = x
            }})
          WITH
            CASE WHEN id(a) < id(b) THEN id(a) ELSE id(b) END AS s,
            CASE WHEN id(a) < id(b) THEN id(b) ELSE id(a) END AS t,
            type(r) AS typ
          WITH DISTINCT s,t,typ
          ORDER BY rand()
          LIMIT $k
          RETURN s,t,typ
        }}
        CALL {{
          WITH s,t,typ
          MATCH (sN) WHERE id(sN)=s
          MATCH (tN) WHERE id(tN)=t
          CREATE (sN)-[:`{reltype}`]->(tN)
          CREATE (tN)-[:`{reltype}`]->(sN)
        }} IN TRANSACTIONS OF $batch ROWS
        RETURN 1
        """
        run(cypher, {"k": limit_k, "ex": exclude_types, "batch": batch_size})

        rows = run_data(f"""
          MATCH ()-[r:`{reltype}`]-()
          WITH id(startNode(r)) AS s, id(endNode(r)) AS t
          WITH CASE WHEN s < t THEN [s,t] ELSE [t,s] END AS p
          RETURN count(DISTINCT p) AS k
        """)
        return int(rows[0]["k"]) if rows else 0

    c_test = _create_pairs(k_test, types["test"], exclude_types=[], batch_size=batch_size)
    exclude = [types["test"]]
    c_val  = 0
    if k_val > 0:
        c_val = _create_pairs(k_val, types["val"], exclude_types=exclude, batch_size=batch_size)
        exclude.append(types["val"])
    c_train = _create_pairs(k_train, types["train"], exclude_types=exclude, batch_size=batch_size)

    return {"P": P, "k_test": c_test, "k_val": c_val, "k_train": c_train}

def _materialize_temp_split_relationships_connected(
    test_holdout: float,
    val_holdout: Optional[float],
    types: _SplitTypes,
    run_suffix: str,
    batch_size: int = 5000
) -> Dict[str, int]:
    """
    Ensure TRAIN is connected by first pinning a spanning forest (backbone)
    into SPLIT_TRAIN_<RUN>, then randomly splitting the remaining edges (non-backbone).
    """
    t = max(0.0, min(1.0, float(test_holdout)))
    v = max(0.0, min(1.0, float(val_holdout or 0.0)))

    row = run_data("""
        MATCH (a)-[r]-(b)
        WHERE NOT type(r) STARTS WITH 'SPLIT_'
        WITH CASE WHEN id(a) < id(b) THEN [id(a), id(b)] ELSE [id(b), id(a)] END AS p
        RETURN count(DISTINCT p) AS P
    """)[0]
    P = int(row["P"]) if row and row.get("P") is not None else 0

    from math import floor
    k_test = int(floor(P * t))
    k_val  = int(floor(P * v))
    k_val  = max(0, min(P - k_test, k_val))
    k_train_quota = max(0, P - k_test - k_val)

    # Build backbone in TRAIN using UNDIRECTED base projection
    base_g = f"_SPLIT_BASE_{run_suffix}"
    g = _project_base_for_split(base_g)
    backbone_pairs: List[Tuple[int,int]] = []
    try:
        if g:
            backbone_pairs = _backbone_edges_with_bfs(g)
    finally:
        if g:
            _drop_in_memory_gds_graph(g)

    c_backbone = _create_train_edges_from_node_ids(backbone_pairs, types["train"], batch_size=batch_size)

    def _create_pairs_excluding_backbone(limit_k: int, reltype: str, exclude_types: List[str]) -> int:
        if limit_k <= 0:
            return 0
        cypher = f"""
        CALL {{
          WITH $k AS k, $ex AS ex
          MATCH (a)-[r]-(b)
          WHERE NOT type(r) STARTS WITH 'SPLIT_'
            AND ALL(x IN ex WHERE NOT EXISTS {{
              MATCH (a)-[r2]-(b) WHERE type(r2) = x
            }})
            AND NOT EXISTS {{
              MATCH (a)-[rT:`{types["train"]}`]-(b)
            }}
          WITH
            CASE WHEN id(a) < id(b) THEN id(a) ELSE id(b) END AS s,
            CASE WHEN id(a) < id(b) THEN id(b) ELSE id(a) END AS t,
            type(r) AS typ
          WITH DISTINCT s,t,typ
          ORDER BY rand()
          LIMIT $k
          RETURN s,t,typ
        }}
        CALL {{
          WITH s,t,typ
          MATCH (sN) WHERE id(sN)=s
          MATCH (tN) WHERE id(tN)=t
          CREATE (sN)-[:`{reltype}`]->(tN)
          CREATE (tN)-[:`{reltype}`]->(sN)
        }} IN TRANSACTIONS OF $batch ROWS
        RETURN 1
        """
        run(cypher, {"k": limit_k, "ex": exclude_types, "batch": batch_size})
        rows = run_data(f"""
          MATCH ()-[r:`{reltype}`]-()
          WITH id(startNode(r)) AS s, id(endNode(r)) AS t
          WITH CASE WHEN s < t THEN [s,t] ELSE [t,s] END AS p
          RETURN count(DISTINCT p) AS k
        """)
        return int(rows[0]["k"]) if rows else 0

    c_test = _create_pairs_excluding_backbone(k_test, types["test"], exclude_types=[])
    exclude = [types["test"]]
    c_val = 0
    if k_val > 0:
        c_val = _create_pairs_excluding_backbone(k_val, types["val"], exclude_types=exclude)
        exclude.append(types["val"])
    remaining_train = max(0, k_train_quota - c_backbone)
    c_train_extra = _create_pairs_excluding_backbone(remaining_train, types["train"], exclude_types=exclude)

    return {
        "P": P,
        "k_backbone": c_backbone,
        "k_test": c_test,
        "k_val": c_val,
        "k_train": c_backbone + c_train_extra,
    }

def _cleanup_temp_types(types: _SplitTypes):
    # Remove only the temp relationships created for this run.
    for t in [types["test"], types["val"], types["train"]]:
        try:
            run(f"MATCH ()-[r:`{t}`]-() DELETE r")
        except Exception:
            pass

# =========================
# Fast stitching via shortest paths on base graph
# =========================

def _wcc_map_for_projected_graph(gname: str) -> Tuple[int, Dict[int, int], Dict[int, List[int]]]:
    """
    Returns (componentCount, { neo4jNodeId -> componentId }, { componentId -> [neo4jNodeId] }) for an existing GDS graph.
    """
    rows = run_data("""
      CALL gds.wcc.stream($g)
      YIELD nodeId, componentId
      WITH gds.util.asNode(nodeId) AS n, componentId
      RETURN id(n) AS nid, componentId
    """, {"g": gname})
    comp_map = {}
    groups: Dict[int, List[int]] = {}
    for r in rows:
        nid = int(r["nid"])
        cid = int(r["componentId"])
        comp_map[nid] = cid
        groups.setdefault(cid, []).append(nid)
    c = len(groups)
    return c, comp_map, groups

def _shortest_path_nodes_on_base_cypher(
    source_neo: int,
    target_neos: List[int],
    max_targets: int = 50,
    max_len: int = 20
) -> List[int]:
    """
    Find a shortest path (by hops) from source_neo to ANY target in target_neos
    using ONLY original relationships (exclude all SPLIT_*). Returns list of Neo IDs.
    Limits the search to a sample of targets (max_targets) and to paths of length <= max_len.
    """
    if not target_neos:
        return []

    # sample targets to keep query cost bounded
    targets = target_neos
    if len(targets) > max_targets:
        # sampling in Python to keep params small
        import random
        targets = random.sample(targets, max_targets)

    # Use shortestPath on an UNDIRECTED topology; exclude SPLIT_* rel-types
    # NOTE: bound the path length to avoid accidental all-graph searches
    rows = run_data(f"""
        MATCH (s) WHERE id(s) = $src
        MATCH (t) WHERE id(t) IN $tgts
        CALL {{
          WITH s, t
          MATCH p = shortestPath( (s)-[r*..{max_len}]-(t) )
          WHERE ALL(r IN relationships(p) WHERE NOT type(r) STARTS WITH 'SPLIT_')
          RETURN p
          ORDER BY length(p) ASC
          LIMIT 1
        }}
        RETURN [n IN nodes(p) | id(n)] AS neoPath
        ORDER BY size(neoPath) ASC
        LIMIT 1
    """, {"src": source_neo, "tgts": targets})

    if not rows:
        return []
    path = rows[0].get("neoPath") or []
    return [int(x) for x in path]


def _connect_train_components_via_paths(
    base_g: str,              
    train_graph_name: str,
    types: _SplitTypes,
    train_label: str,
    max_targets_per_comp: int = 50
):
    comp_count, comp_map, groups = _wcc_map_for_projected_graph(train_graph_name)
    if comp_count <= 1:
        return

    # largest (main) component by node count
    main_cid = max(groups.keys(), key=lambda cid: len(groups[cid]))
    main_neos = groups[main_cid]

    for cid, nodes in groups.items():
        if cid == main_cid:
            continue

        rep = nodes[0]

        # shortest path on original, non-SPLIT topology (Cypher), returns **Neo IDs**
        path_neos = _shortest_path_nodes_on_base_cypher(
            source_neo=rep,
            target_neos=main_neos,
            max_targets=max_targets_per_comp,
            max_len=20,  
        )

        if not path_neos or len(path_neos) < 2:
            # fallback with larger sample and/or longer bound
            path_neos = _shortest_path_nodes_on_base_cypher(
                source_neo=rep,
                target_neos=main_neos,
                max_targets=min(len(main_neos), 200),
                max_len=40,
            )
            if not path_neos or len(path_neos) < 2:
                # give up on this component (should be rare if the original graph is weakly connected)
                continue

        # add mirrored TRAIN edges for every consecutive pair in the path
        pairs = []
        for i in range(len(path_neos) - 1):
            u = path_neos[i]; v = path_neos[i+1]
            if u == v:
                continue
            if u > v:
                u, v = v, u
            pairs.append([u, v])

        if pairs:
            # only add TRAIN edges on pairs that have no SPLIT_* already
            run(f"""
                UNWIND $pairs AS p
                MATCH (a) WHERE id(a)=p[0]
                MATCH (b) WHERE id(b)=p[1]
                WITH a, b
                WHERE NOT EXISTS {{ MATCH (a)-[r]-(b) WHERE type(r) STARTS WITH 'SPLIT_' }}
                MERGE (a)-[:`{types["train"]}`]->(b)
                MERGE (b)-[:`{types["train"]}`]->(a)
            """, {"pairs": pairs})


        # ensure endpoints are tagged so the train projection includes them
        run(f"""
          UNWIND $ids AS i
          MATCH (n) WHERE id(n)=i
          SET n:`{_safe_label(train_label)}`
        """, {"ids": path_neos})

def _tag_all_original_nodes(temp_label: str):
    temp_label = _safe_label(temp_label)
    run(f"""
      MATCH (n)
      SET n:`{temp_label}`
    """)


# =========================
# Connectivity helper (optional)
# =========================
def _train_connectivity(name: str) -> Tuple[Optional[bool], Optional[int]]:
    try:
        rows = run_data("CALL gds.wcc.stats($g) YIELD componentCount RETURN componentCount", {"g": name})
        if rows and "componentCount" in rows[0]:
            comp = int(rows[0]["componentCount"])
            return (comp == 1, comp)
    except Exception:
        pass
    return (None, None)

# =========================
# Request model
# =========================
class ExecuteSplitBody(BaseModel):
    trainGraphName: str = "trainGraph"
    testGraphName: str = "testGraph"
    valGraphName: Optional[str] = "valGraph"  
    testHoldout: float = Field(default=0.10, ge=0.0, le=1.0)
    valHoldout: float = Field(default=0.10, ge=0.0, le=1.0)
    ensureConnected: bool = True  # ensure TRAIN is connected via backbone + path stitching
    includeEmbeddingProps: bool = False  
    reFreshFeatureCache: bool = True     # refresh feature cache each run

# =========================
# Route
# =========================
@router.post("/execute")
def execute_master_split(body: ExecuteSplitBody):
    run_suffix = _mk_run_suffix()
    types = _build_run_types(run_suffix)

    # Load & filter node properties for projection
    raw_props = ensure_feature_props_loaded(force=body.reFreshFeatureCache)
    props = _filter_supported_node_props(raw_props)
    # exclude embedding-like props unless explicitly allowed
    if not body.includeEmbeddingProps:
        props = _drop_embedding_props(props)

    # Optional VAL creation
    want_val = bool(body.valGraphName) and (body.valHoldout > 0.0)

    # Create unique, safe endpoint labels per slice
    train_ep_label = f"__SPLIT_TRAIN_EP_{run_suffix}__"
    test_ep_label  = f"__SPLIT_TEST_EP_{run_suffix}__"
    val_ep_label   = f"__SPLIT_VAL_EP_{run_suffix}__"

    # We'll keep a base UNDIRECTED projection alive during the run for fast paths
    base_g = f"_SPLIT_BASE_{run_suffix}"

    try:
        # 1) Materialize temporary split relationships (connected or random)
        if body.ensureConnected:
            counts = _materialize_temp_split_relationships_connected(
                test_holdout=body.testHoldout,
                val_holdout=body.valHoldout if want_val else 0.0,
                types=types,
                run_suffix=run_suffix
            )
        else:
            counts = _materialize_temp_split_relationships(
                test_holdout=body.testHoldout,
                val_holdout=body.valHoldout if want_val else 0.0,
                types=types
            )

        # 1) Tag all ORIGINAL nodes so they appear as nodes in the TRAIN projection
        _tag_all_original_nodes(train_ep_label)

        # 2) Also tag TRAIN endpoints
        _tag_split_endpoints(train_ep_label, types["train"])

        train_info = _project_with_label_only(
            body.trainGraphName, train_ep_label, [types["train"]], props
        )


        if body.ensureConnected:
            # 2b) Build UNDIRECTED base projection once, then stitch components via shortest paths
            g = _project_base_for_split(base_g)
            if g:
                _connect_train_components_via_paths(
                    base_g, body.trainGraphName, types, train_ep_label, max_targets_per_comp=50
                )
                # Re-project Train once after stitching
                _tag_split_endpoints(train_ep_label, types["train"])
                train_info = _project_with_label_only(
                    body.trainGraphName, train_ep_label, [types["train"]], props
                )

        # TEST
        _tag_split_endpoints(test_ep_label, types["test"])
        test_info = _project_with_label_only(
            body.testGraphName, test_ep_label, [types["test"]], props
        )

        # VAL (optional)
        val_info = None
        if want_val:
            _tag_split_endpoints(val_ep_label, types["val"])
            val_info = _project_with_label_only(
                body.valGraphName or "valGraph", val_ep_label, [types["val"]], props
            )

        # 3) (Optional) connectivity on train
        connected, components = _train_connectivity(body.trainGraphName)

        return {
            "ok": True,
            "counts": counts,  # undirected pairs per split (measured)
            "train": train_info,
            "test": test_info,
            "validation": val_info,
            "testHoldout": body.testHoldout,
            "valHoldout": body.valHoldout if want_val else 0.0,
            "trainConnected": connected,
            "trainComponents": components,
            "featurePropsRequested": raw_props,
            "featurePropsUsed": props,
            "runSuffix": run_suffix
        }

    finally:
        # 4) Always clean up the temporary relationships and projections we created for this run
        _cleanup_temp_types(types)
        # Remove temporary endpoint labels
        for lbl in (train_ep_label, test_ep_label, val_ep_label):
            try:
                _untag_split_endpoints(lbl)
            except Exception:
                pass
        # Drop base projection if it exists
        try:
            _drop_in_memory_gds_graph(base_g)
        except Exception:
            pass
