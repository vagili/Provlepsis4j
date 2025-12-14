# app/routers/string_feature_convert.py
from typing import Any, Dict, List, Optional
from fastapi import HTTPException
from ..db import run, run_data

# Properties we never treat as features
EXCLUDE_KEYS = ["Node2Vec", "FastRP", "GraphSAGE", "HashGNN"]

def _db_total_nodes() -> int:
    rows = run_data("MATCH (n) RETURN count(n) AS N")
    return int(rows[0]["N"]) if rows else 0

def _discover_full_coverage_string_keys(coverage: float) -> List[str]:
    """
    Return node property keys that:
      * are not in EXCLUDE_KEYS,
      * exist on at least coverage * N nodes,
      * and whose non-null values are *all* strings.
    Uses a safe test: all(v IN vals WHERE toString(v) = v).
    """
    cypher = """
    MATCH (n) WITH count(n) AS N
    CALL db.propertyKeys() YIELD propertyKey AS k
    WITH N, k
    WHERE NOT k IN $exclude
      AND NOT k STARTS WITH 'numeric_'

    // gather non-null samples for coverage + type test
    MATCH (m) WHERE m[k] IS NOT NULL
    WITH N, k, collect(m[k]) AS vals, count(m) AS present,
         toInteger(ceil($coverage * N)) AS cov_min
    WHERE present >= cov_min
      AND all(v IN vals WHERE toString(v) = v)

    RETURN collect(k) AS keys
    """
    rows = run_data(cypher, {"exclude": EXCLUDE_KEYS, "coverage": float(coverage)})
    return rows[0]["keys"] if rows else []


def _categories_for_key(key: str, max_categories: int) -> List[str]:
    """
    Distinct scalar string values for the given key (cut off at max_categories+1).
    """
    rows = run_data(
        """
        WITH $k AS k
        MATCH (n)
        WITH n[k] AS v WHERE v IS NOT NULL
        WITH DISTINCT v LIMIT $limit
        RETURN collect(v) AS cats
        """,
        {"k": key, "limit": max_categories + 1},
    )
    cats: List[str] = rows[0]["cats"] if rows else []
    if len(cats) > max_categories:
        return []
    return sorted(cats)

def _slug(s: str) -> str:
    out = []
    for ch in str(s):
        if ch.isalnum():
            out.append(ch)
        elif ch in "- ":
            out.append("_")
        else:
            out.append("_")
    slug = "".join(out).strip("_")
    return slug or "val"

def _write_one_hot_scalar_for_key(key: str, categories: List[str]) -> int:
    """
    For each category, create a scalar property numeric_<key>_<slug(cat)> = 1.0/0.0.
    Returns the total number of node property assignments across all categories.
    """
    total_set = 0
    for cat in categories:
        prop = f"numeric_{key}_{_slug(cat)}"
        cypher = f"""
        WITH $k AS k, $cat AS cat
        MATCH (n) WHERE n[k] IS NOT NULL
        SET n.`{prop}` = CASE WHEN n[k] = cat THEN 1.0 ELSE 0.0 END
        RETURN count(n) AS c
        """
        rows = run_data(cypher, {"k": key, "cat": cat})
        total_set += int(rows[0]["c"]) if rows else 0
    return total_set

def _drop_original_keys(keys: List[str]) -> Dict[str, Any]:
    if not keys:
        return {"droppedKeys": [], "totalDroppedAssignments": 0}
    # APOC path
    rows = run_data(
        """
        MATCH (n)
        WITH n, $keys AS keys
        WITH n, [k IN keys WHERE n[k] IS NOT NULL] AS toDrop
        WHERE size(toDrop) > 0
        CALL apoc.create.removeProperties(n, toDrop) YIELD node
        RETURN count(node) AS c
        """,
        {"keys": keys},
    )
    if rows is None:
        # APOC not installed; skip silently
        return {"droppedKeys": [], "totalDroppedAssignments": 0, "note": "APOC not available; original strings kept."}
    return {"droppedKeys": keys, "totalDroppedAssignments": int(rows[0]["c"])}

def _purge_any_stringish_props() -> Dict[str, Any]:
    """
    Remove any remaining node/relationship properties that contain non-numeric
    (and non-bool) values (to avoid GDS projection issues).
    """
    def _find_node_stringish() -> List[str]:
        rows = run_data(
            """
            CALL db.propertyKeys() YIELD propertyKey AS k
            WITH [k IN collect(k) WHERE NOT k IN $exclude AND NOT k STARTS WITH 'numeric_'] AS cand
            UNWIND cand AS k
            CALL {
              WITH k
              MATCH (n)
              WHERE n[k] IS NOT NULL
                AND (
                  toFloatOrNull(n[k]) IS NULL     // not numeric
                  AND NOT n[k] IN [true,false]    // not bool
                )
              RETURN 1 AS bad LIMIT 1
            }
            WITH k, bad WHERE bad = 1
            RETURN collect(k) AS keys
            """,
            {"exclude": EXCLUDE_KEYS},
        )
        return rows[0]["keys"] if rows else []

    def _find_rel_stringish() -> List[str]:
        rows = run_data(
            """
            MATCH ()-[r]-()
            WITH DISTINCT keys(r) AS ks
            UNWIND ks AS k
            WITH DISTINCT k WHERE NOT k STARTS WITH 'numeric_' AND NOT k IN $exclude
            CALL {
              WITH k
              MATCH ()-[r]-()
              WHERE r[k] IS NOT NULL
                AND (
                  toFloatOrNull(r[k]) IS NULL
                  AND NOT r[k] IN [true,false]
                )
              RETURN 1 AS bad LIMIT 1
            }
            WITH k, bad WHERE bad = 1
            RETURN collect(k) AS keys
            """,
            {"exclude": EXCLUDE_KEYS},
        )
        return rows[0]["keys"] if rows else []

    node_keys = _find_node_stringish()
    rel_keys  = _find_rel_stringish()

    node_count = 0
    rel_count = 0

    if node_keys:
        rows = run_data(
            """
            MATCH (n)
            WITH n, $keys AS keys
            WITH n, [k IN keys WHERE n[k] IS NOT NULL] AS toDrop
            WHERE size(toDrop) > 0
            CALL apoc.create.removeProperties(n, toDrop) YIELD node
            RETURN count(node) AS c
            """,
            {"keys": node_keys},
        )
        if rows is not None:
            node_count = int(rows[0]["c"])

    if rel_keys:
        rows = run_data(
            """
            MATCH ()-[r]-()
            WITH r, $keys AS keys
            WITH r, [k IN keys WHERE r[k] IS NOT NULL] AS toDrop
            WHERE size(toDrop) > 0
            CALL apoc.create.removeProperties(r, toDrop) YIELD rel
            RETURN count(rel) AS c
            """,
            {"keys": rel_keys},
        )
        if rows is not None:
            rel_count = int(rows[0]["c"])

    return {
        "nodeKeysPurged": node_keys,
        "relKeysPurged": rel_keys,
        "nodeAssignmentsDropped": node_count,
        "relAssignmentsDropped": rel_count
    }

def convert_strings_best_effort(
    coverage: float = 1.0,
    max_categories: int = 256,
    drop_original: bool = True,
    purge_rest: bool = True,
) -> Dict[str, Any]:
    """
    Convert full-coverage scalar string node props into *scalar* one-hot features:
      numeric_<key>_<category> ∈ {0.0,1.0}
    Skip if too many categories, skip partial-coverage keys,
    then optionally drop originals and purge any leftover string-ish props.
    """
    total_nodes = _db_total_nodes()
    if total_nodes == 0:
        return {"ok": True, "message": "No nodes in DB.", "converted": [], "totalNodes": 0}

    keys = _discover_full_coverage_string_keys(coverage=coverage)
    converted: List[Dict[str, Any]] = []
    for k in keys:
        cats = _categories_for_key(k, max_categories=max_categories)
        if not cats:
            converted.append({"key": k, "status": "skipped-too-many-or-empty"})
            continue

        written = _write_one_hot_scalar_for_key(k, cats)
        converted.append({
            "key": k,
            "status": "converted",
            "numCategories": len(cats),
            "assignmentsWritten": written
        })

    drop_summary: Optional[Dict[str, Any]] = None
    if drop_original and converted:
        drop_summary = _drop_original_keys([x["key"] for x in converted if x["status"] == "converted"])

    purge_summary: Optional[Dict[str, Any]] = None
    if purge_rest:
        purge_summary = _purge_any_stringish_props()

    return {
        "ok": True,
        "totalNodes": total_nodes,
        "converted": converted,
        "droppedOriginals": drop_summary or {"droppedKeys": [], "totalDroppedAssignments": 0},
        "purgedStringish": purge_summary or {
            "nodeKeysPurged": [], "relKeysPurged": [],
            "nodeAssignmentsDropped": 0, "relAssignmentsDropped": 0
        }
    }
