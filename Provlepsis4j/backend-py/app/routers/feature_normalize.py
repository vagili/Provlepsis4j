# app/logic/feature_normalize.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
import re

from ..db import run, run_data

# Properties we never touch or use as features
EXCLUDE_KEYS = {"Node2Vec", "FastRP", "GraphSAGE", "HashGNN"}
SAFE_PROP = re.compile(r"^[A-Za-z0-9_]+$")

def _with_db(db: str, cypher: str) -> str:
    return f"USE `{db}`\n{cypher}"

def _safe_prop_or_raise(key: str) -> str:
    if not SAFE_PROP.match(key):
        raise ValueError(f"invalid property key: {key!r}")
    return key

def _db_total_nodes(db: str) -> int:
    rows = run_data(_with_db(db, "MATCH (n) RETURN count(n) AS N"))
    return int(rows[0]["N"]) if rows else 0

def _present_count(db: str, key: str) -> int:
    rows = run_data(
        _with_db(db, "WITH $k AS k MATCH (n) WHERE n[k] IS NOT NULL RETURN count(n) AS present"),
        {"k": key},
    )
    return int(rows[0]["present"]) if rows else 0

def _values_for_key(db: str, key: str) -> List[Any]:
    rows = run_data(
        _with_db(db, """
        WITH $k AS k
        MATCH (n)
        WITH n[k] AS v
        WHERE v IS NOT NULL
        RETURN v
        """),
        {"k": key},
    )
    return [r["v"] for r in rows]

def _rows_id_and_value(db: str, key: str) -> List[Dict[str, Any]]:
    return run_data(
        _with_db(db, """
        WITH $k AS k
        MATCH (n)
        WITH id(n) AS id, n[k] AS v
        WHERE v IS NOT NULL
        RETURN id, v
        ORDER BY id
        """),
        {"k": key},
    )

def _write_scalar_property(db: str, key: str, rows: List[Dict[str, Any]]) -> int:
    """
    SET a scalar numeric property for the rows (id, val).
    """
    cypher = f"""
    UNWIND $rows AS row
    MATCH (n) WHERE id(n) = row.id
    SET n.`{_safe_prop_or_raise(key)}` = row.val
    RETURN count(*) AS written
    """
    out = run_data(_with_db(db, cypher), {"rows": rows})
    return int(out[0]["written"]) if out else 0

def _drop_original_keys(db: str, keys: List[str]) -> int:
    """
    Drops original string keys from nodes using APOC.
    """
    if not keys:
        return 0
    rows = run_data(
        _with_db(db, """
        MATCH (n)
        WITH n, $keys AS keys
        WITH n, [k IN keys WHERE n[k] IS NOT NULL] AS toDrop
        WHERE size(toDrop) > 0
        CALL apoc.create.removeProperties(n, toDrop) YIELD node
        RETURN count(node) AS c
        """),
        {"keys": keys},
    )
    return int(rows[0]["c"]) if rows else 0

def _purge_leftover_stringish(db: str) -> Dict[str, int]:
    """
    Remove *any* remaining node/relationship properties that contain non-numeric values.
    Keeps numeric_* props, and doesn't touch EXCLUDE_KEYS.
    """
    node_keys = run_data(_with_db(db, """
    CALL db.propertyKeys() YIELD propertyKey AS k
    WITH [k IN collect(k) WHERE NOT k IN $exclude AND NOT k STARTS WITH 'numeric_'] AS cand
    UNWIND cand AS k
    CALL {
      WITH k
      MATCH (n)
      WHERE n[k] IS NOT NULL AND toFloatOrNull(n[k]) IS NULL
      RETURN 1 AS bad LIMIT 1
    }
    WITH k, bad WHERE bad = 1
    RETURN collect(k) AS keys
    """), {"exclude": list(EXCLUDE_KEYS)})
    node_keys = node_keys[0]["keys"] if node_keys else []

    rel_keys = run_data(_with_db(db, """
    MATCH ()-[r]-()
    WITH DISTINCT keys(r) AS ks
    UNWIND ks AS k
    WITH DISTINCT k WHERE NOT k STARTS WITH 'numeric_' AND NOT k IN $exclude
    CALL {
      WITH k
      MATCH ()-[r]-()
      WHERE r[k] IS NOT NULL AND toFloatOrNull(r[k]) IS NULL
      RETURN 1 AS bad LIMIT 1
    }
    WITH k, bad WHERE bad = 1
    RETURN collect(k) AS keys
    """), {"exclude": list(EXCLUDE_KEYS)})
    rel_keys = rel_keys[0]["keys"] if rel_keys else []

    node_dropped = 0
    rel_dropped = 0

    if node_keys:
        rows = run_data(_with_db(db, """
        MATCH (n)
        WITH n, $keys AS keys
        WITH n, [k IN keys WHERE n[k] IS NOT NULL] AS toDrop
        WHERE size(toDrop) > 0
        CALL apoc.create.removeProperties(n, toDrop) YIELD node
        RETURN count(node) AS c
        """), {"keys": node_keys})
        node_dropped = int(rows[0]["c"]) if rows else 0

    if rel_keys:
        rows = run_data(_with_db(db, """
        MATCH ()-[r]-()
        WITH r, $keys AS keys
        WITH r, [k IN keys WHERE r[k] IS NOT NULL] AS toDrop
        WHERE size(toDrop) > 0
        CALL apoc.create.removeProperties(r, toDrop) YIELD rel
        RETURN count(rel) AS c
        """), {"keys": rel_keys})
        rel_dropped = int(rows[0]["c"]) if rows else 0

    return {"nodeAssignmentsDropped": node_dropped, "relAssignmentsDropped": rel_dropped}

def _sanitize_suffix(s: str) -> str:
    # keep simple and readable
    s = str(s).strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_]", "_", s)
    if not s:
        s = "cat"
    return s[:64]  # keep key lengths reasonable

def _chunk(lst: List[Any], size: int):
    for i in range(0, len(lst), size):
        yield lst[i:i+size]

def normalize_features_for_db(
    db: str,
    *,
    coverage: float = 1.0,
    max_categories: int = 256,
    one_hot_prefix: str = "numeric_",
    drop_original: bool = True,
    purge_leftovers: bool = True,
    batch_size: int = 1000,
) -> Dict[str, Any]:
    """
    Convert *full-coverage* scalar string node properties into separate numeric one-hot scalars:
      key='color' with categories {'red','blue'}  ->  numeric_color__red, numeric_color__blue
    Leaves already-numeric props as-is.
    Skips any key that:
      - is missing on any node (full coverage required at given threshold)
      - has > max_categories distinct values
      - contains any non-string values
      - is in EXCLUDE_KEYS or starts with 'numeric_'
    """
    N = _db_total_nodes(db)
    if N == 0:
        return {"ok": True, "converted": [], "note": "No nodes in DB."}

    # Candidate keys = present on >= coverage * N nodes AND not in exclude/prefix
    rows = run_data(_with_db(db, """
    MATCH (n) WITH count(n) AS N
    CALL db.propertyKeys() YIELD propertyKey AS k
    WITH N, k WHERE NOT k IN $exclude AND NOT k STARTS WITH 'numeric_'
    MATCH (m)
    WHERE m[k] IS NOT NULL
    WITH N, k, count(m) AS present
    WITH N, k, present, toInteger(ceil($coverage * N)) AS cov_min
    WHERE present >= cov_min
    RETURN collect(k) AS keys
    """), {"exclude": list(EXCLUDE_KEYS), "coverage": float(coverage)})
    cand_keys = rows[0]["keys"] if rows else []
    if not cand_keys:
        out = {"ok": True, "converted": [], "note": "No keys met coverage threshold."}
        if purge_leftovers:
            out["purge"] = _purge_leftover_stringish(db)
        return out

    converted: List[Dict[str, Any]] = []
    to_drop: List[str] = []

    for key in sorted(cand_keys):
        # double-check coverage == N if you asked for 1.0
        if coverage >= 1.0 and _present_count(db, key) != N:
            continue

        # fetch sample values and confirm all are strings
        vals = _values_for_key(db, key)
        if not vals:
            continue
        if any(not isinstance(v, str) for v in vals):
            # mixed or non-string -> skip
            continue

        cats = sorted(set(vals))
        if len(cats) == 0:
            continue
        if len(cats) > max_categories:
            continue

        # construct one-hot scalar property names
        base = f"{one_hot_prefix}{key}"
        _safe_prop_or_raise(base)  # base is validated, suffixes sanitized below
        cat_props = [f"{base}__{_sanitize_suffix(c)}" for c in cats]
        for p in cat_props:
            _safe_prop_or_raise(p)

        # get (id, value) rows (full coverage guaranteed)
        node_rows = _rows_id_and_value(db, key)

        # For each category, write 0/1 scalar
        total_writes = 0
        for cat, prop_name in zip(cats, cat_props):
            rows_01 = [{"id": r["id"], "val": 1.0 if r["v"] == cat else 0.0} for r in node_rows]
            for ch in _chunk(rows_01, batch_size):
                total_writes += _write_scalar_property(db, prop_name, ch)

        converted.append({
            "key": key,
            "numCategories": len(cats),
            "createdProps": cat_props,
            "nodeAssignmentsWritten": total_writes
        })
        to_drop.append(key)

    drop_report: Optional[int] = None
    if drop_original and to_drop:
        drop_report = _drop_original_keys(db, to_drop)

    out: Dict[str, Any] = {
        "ok": True,
        "coverage": coverage,
        "maxCategories": max_categories,
        "converted": converted,
        "droppedOriginalCount": drop_report if drop_report is not None else 0,
    }
    if purge_leftovers:
        out["purge"] = _purge_leftover_stringish(db)
    return out
