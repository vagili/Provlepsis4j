# app/routers/load_graph.py
from fastapi import APIRouter, Form, UploadFile, File, HTTPException, Body
from typing import Optional, Dict, Any, List, Tuple
import pandas as pd
import io
import re
import numpy as np

from ..db import run, run_data
router = APIRouter(prefix="/load")

# ---------- tiny helpers for DB selection ----------

def _with_db(db: str, cypher: str) -> str:
    return f"USE `{db}`\n{cypher}"

def _create_new_database(prefix: str = "dataset", explicit: str | None = None) -> str:
    rows = run_data("SHOW DATABASES YIELD name RETURN collect(name) AS names")
    existing = set(rows[0].get("names", [])) if rows else set()

    # If the user provided an explicit name, use that instead of the incremental one
    if explicit:
        candidate = explicit.strip()
        if not candidate:
            raise HTTPException(status_code=400, detail="Database name cannot be empty.")
        candidate = candidate.lower()
        if candidate in existing:
            raise HTTPException(
                status_code=400,
                detail=f"Database '{candidate}' already exists. Choose a different name.",
            )
        run(f"CREATE DATABASE `{candidate}` IF NOT EXISTS WAIT")
        return candidate

    pattern = re.compile(rf"^{re.escape(prefix)}-(\d+)$")
    max_n = 0
    for name in existing:
        m = pattern.match(name)
        if m:
            try:
                max_n = max(max_n, int(m.group(1)))
            except ValueError:
                pass

    n = max_n + 1
    while True:
        candidate = f"{prefix}-{n}".lower()
        if candidate not in existing:
            break
        n += 1

    run(f"CREATE DATABASE `{candidate}` IF NOT EXISTS WAIT")
    return candidate

# ---------- CSV helpers ----------

def _read_csv(upload: UploadFile) -> pd.DataFrame:
    data = upload.file.read()
    upload.file.close()
    return pd.read_csv(io.BytesIO(data))

def _sanitize_type(s: str) -> str:
    s = "".join(c for c in str(s) if c.isalnum() or c == "_")
    return s or "REL"

def _chunk(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i+size]

def _mk_constraint_entity(db: str):
    run(_with_db(db, "CREATE CONSTRAINT IF NOT EXISTS FOR (n:`Entity`) REQUIRE n.id IS UNIQUE"))

def _rows_for_nodes(ids: List[Any]) -> List[Dict[str, Any]]:
    return [{"id": i} for i in ids]

def _map_rows_for_props(df: pd.DataFrame, id_col: str) -> List[Dict[str, Any]]:
    rows = []
    for _, r in df.iterrows():
        props = r.drop(labels=[id_col], errors="ignore").to_dict()
        rows.append({"id": r[id_col], "props": props})
    return rows

def _detect_edges_columns(df: pd.DataFrame) -> Tuple[str, str, Optional[str]]:
    cols = {c.lower(): c for c in df.columns}
    src = cols.get("source") or cols.get("src")
    dst = cols.get("target") or cols.get("dst")
    if not src or not dst:
        raise HTTPException(
            status_code=400,
            detail="Edges CSV must contain 'source' and 'target' (or 'src'/'dst') columns."
        )
    typ = cols.get("type") or cols.get("rel_type")
    return src, dst, typ

# ---------- Cypher templates ----------

def _merge_nodes_query() -> str:
    return """
    UNWIND $rows AS row
    MERGE (n:`Entity` {id: row.id})
    """

def _set_props_query() -> str:
    return """
    UNWIND $rows AS row
    MERGE (n:`Entity` {id: row.id})
    SET n += row.props
    """

def _merge_rels_query(rel_type: str) -> str:
    return f"""
    UNWIND $rows AS row
    MATCH (a:`Entity` {{id: row.src}}), (b:`Entity` {{id: row.dst}})
    MERGE (a)-[r:`{rel_type}`]->(b)
    SET r += row.props
    SET r.predicted = 0
    """

# =====================================================================
# Optional cleanup (purge string-ish props via APOC)
# =====================================================================

EXCLUDE_KEYS = ["Node2Vec", "FastRP", "GraphSAGE", "HashGNN"]

def _find_node_stringish_keys(db: str) -> List[str]:
    cypher = """
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
    """
    rows = run_data(_with_db(db, cypher), {"exclude": EXCLUDE_KEYS})
    return rows[0]["keys"] if rows else []

def _find_rel_stringish_keys(db: str) -> List[str]:
    cypher = """
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
    """
    rows = run_data(_with_db(db, cypher), {"exclude": EXCLUDE_KEYS})
    return rows[0]["keys"] if rows else []

def _purge_stringish_props(db: str) -> Dict[str, Any]:
    node_keys = _find_node_stringish_keys(db)
    rel_keys  = _find_rel_stringish_keys(db)

    node_count = 0
    rel_count = 0

    if node_keys:
        rows = run_data(
            _with_db(db, """
            MATCH (n)
            WITH n, $keys AS keys
            WITH n, [k IN keys WHERE n[k] IS NOT NULL] AS toDrop
            WHERE size(toDrop) > 0
            CALL apoc.create.removeProperties(n, toDrop) YIELD node
            RETURN count(node) AS c
            """),
            {"keys": node_keys},
        )
        if rows is None:
            raise HTTPException(status_code=500, detail="APOC not available. Install APOC or disable purge_all_stringish.")
        node_count = int(rows[0]["c"])

    if rel_keys:
        rows = run_data(
            _with_db(db, """
            MATCH ()-[r]-()
            WITH r, $keys AS keys
            WITH r, [k IN keys WHERE r[k] IS NOT NULL] AS toDrop
            WHERE size(toDrop) > 0
            CALL apoc.create.removeProperties(r, toDrop) YIELD rel
            RETURN count(rel) AS c
            """),
            {"keys": rel_keys},
        )
        if rows is None:
            raise HTTPException(status_code=500, detail="APOC not available. Install APOC or disable purge_all_stringish.")
        rel_count = int(rows[0]["c"])

    return {
        "nodeKeysPurged": node_keys,
        "relKeysPurged": rel_keys,
        "nodeAssignmentsDropped": node_count,
        "relAssignmentsDropped": rel_count
    }

# =====================================================================
# In-memory feature transformation (scalar one-hot, no arrays)
# =====================================================================

_SAFE_SUFFIX = re.compile(r"[^A-Za-z0-9_]+")

def _safe_cat_suffix(value: Any) -> str:
    """Sanitize category value into a safe suffix."""
    s = str(value)
    s = _SAFE_SUFFIX.sub("_", s).strip("_")
    return s or "val"

def _transform_feature_df(
    df: pd.DataFrame,
    id_col: str,
    max_categories: int = 256
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Rules:
      - Keep numeric columns as-is (coerce dtype).
      - If a column is all-strings AND has NO nulls -> expand into scalar one-hot columns:
            numeric_<col>_<cat> = 0.0/1.0
        and DROP the original column.
      - If a string column has ANY nulls -> DROP column entirely (no encoding).
      - If mixed (neither numeric nor all-strings) -> DROP column.
    """
    out = df.copy()
    summary: Dict[str, Any] = {
        "kept_numeric": [],
        "converted_one_hot_scalar": [],
        "dropped_string_with_nulls": [],
        "dropped_mixed": [],
        "skipped_too_many_categories": [],
    }

    candidate_cols = [c for c in out.columns if c != id_col]

    for c in candidate_cols:
        if c not in out.columns:
            # might have been dropped/expanded already
            continue

        col = out[c]
        non_null = col.dropna()
        has_nulls = len(non_null) != len(col)

        if len(non_null) == 0:
            # completely empty -> drop
            out.drop(columns=[c], inplace=True)
            summary["dropped_mixed"].append(c)
            continue

        # 1) Numeric-ish? keep (even if some nulls)
        can_be_numeric = pd.to_numeric(non_null, errors="coerce").notna().all()
        if can_be_numeric:
            out[c] = pd.to_numeric(out[c], errors="coerce")
            summary["kept_numeric"].append(c)
            continue

        # 2) All strings
        all_strings = non_null.apply(lambda v: isinstance(v, str)).all()
        if all_strings:
            if has_nulls:
                # We do NOT encode partial-coverage string columns; drop to avoid conflicts
                out.drop(columns=[c], inplace=True)
                summary["dropped_string_with_nulls"].append(c)
                continue

            # --- on reruns, drop any existing numeric_<c>_* columns so we overwrite ---
            numeric_prefix = f"numeric_{c}_"
            existing_numeric_cols = [
                colname for colname in out.columns
                if colname.startswith(numeric_prefix)
            ]
            if existing_numeric_cols:
                out.drop(columns=existing_numeric_cols, inplace=True)

            cats = sorted(non_null.unique().tolist())
            if len(cats) == 0:
                out.drop(columns=[c], inplace=True)
                summary["dropped_mixed"].append(c)
                continue
            if len(cats) > max_categories:
                # too many categories -> drop column rather than explode property count
                out.drop(columns=[c], inplace=True)
                summary["skipped_too_many_categories"].append({
                    "column": c,
                    "numCategories": len(cats),
                })
                continue

            # Build safe unique column names for each category
            used = set()

            cat_to_col: Dict[Any, str] = {}
            for i, cat in enumerate(cats):
                base = _safe_cat_suffix(cat)
                name = f"numeric_{c}_{base}"
                # ensure uniqueness if collision
                j = 1
                candidate = name
                while candidate in used or candidate in out.columns:
                    j += 1
                    candidate = f"{name}_{j}"
                used.add(candidate)
                cat_to_col[cat] = candidate

            # Create each one-hot scalar column
            for cat, new_col in cat_to_col.items():
                out[new_col] = (out[c] == cat).astype(float)

            # Drop original
            out.drop(columns=[c], inplace=True)

            summary["converted_one_hot_scalar"].append({
                "column": c,
                "created": list(cat_to_col.values()),
                "numCategories": len(cats)
            })
            continue

        # 3) Mixed or other types → drop
        out.drop(columns=[c], inplace=True)
        summary["dropped_mixed"].append(c)

    return out, summary

# ---------- endpoint ----------
@router.post("/normalize-existing")
def normalize_existing_db(
    db: str = Body(..., embed=True),
    batch_size: int = 20000,
    max_categories: int = 256,
    purge_all_stringish: bool = True,
):
    """
    Normalize features for an *existing* dataset DB.

    1) Read :Entity nodes + their properties from `db`.
    2) Run the same in-memory _transform_feature_df as used for CSV import
       (numeric kept, string -> numeric_<col>_<cat>, mixed dropped).
    3) Write transformed features back to the same nodes.
    4) Optionally purge any remaining string-ish props (nodes & rels) via APOC.
    """
    # 1) Pull all :Entity nodes into a DataFrame
    rows = run_data(
        _with_db(
            db,
            "MATCH (n) RETURN n.id AS id, properties(n) AS props",
        )
    )

    if not rows:
        return {
            "status": "ok",
            "database": db,
            "feature_rows_processed": 0,
            "feature_transform_summary": {
                "kept_numeric": [],
                "converted_one_hot_scalar": [],
                "dropped_string_with_nulls": [],
                "dropped_mixed": [],
                "skipped_too_many_categories": [],
            },
            "purged_stringish_properties": {
                "nodeKeysPurged": [],
                "relKeysPurged": [],
                "nodeAssignmentsDropped": 0,
                "relAssignmentsDropped": 0,
            },
            "note": "No :Entity nodes found.",
        }

    records: list[dict[str, Any]] = []
    for r in rows:
        props = dict(r.get("props") or {})
        rec = {"id": r["id"]}
        rec.update(props)
        records.append(rec)

    df_raw = pd.DataFrame.from_records(records)

    # 2) Run the same transformer used by /load/graph
    df_transformed, feature_transform_summary = _transform_feature_df(
        df_raw, id_col="id", max_categories=max_categories
    )

    # 3) Write back transformed props (numeric + one-hot)
    feat_rows = _map_rows_for_props(df_transformed, id_col="id")
    feat_total = len(feat_rows)

    for chunk in _chunk(feat_rows, batch_size):
        run(_with_db(db, _set_props_query()), {"rows": chunk})

    # 4) Drop any remaining string-ish props (nodes + rels)
    purge_summary = _purge_stringish_props(db) if purge_all_stringish else {
        "nodeKeysPurged": [],
        "relKeysPurged": [],
        "nodeAssignmentsDropped": 0,
        "relAssignmentsDropped": 0,
    }

    return {
        "status": "ok",
        "database": db,
        "feature_rows_processed": feat_total,
        "feature_transform_summary": feature_transform_summary,
        "purged_stringish_properties": purge_summary,
    }

@router.post("/graph")
async def load_graph(
    edges: UploadFile = File(...),
    features: Optional[UploadFile] = File(None),
    batch_size: int = 20000,
    max_categories: int = 256,
    purge_all_stringish: bool = True,
    isTemporal: bool = Form(False),
    timestampColumn: str | None = Form(None),
    dataset_name: str | None = Form(None),
):
    """
    1) Create a fresh DB and load edges (+ features if provided).
    2) Transform features in-memory:
         - numeric kept
         - clean strings -> scalar one-hot columns (numeric_<col>_<cat>)
         - string-with-nulls dropped
         - mixed dropped
    3) Optionally purge any remaining string-ish props (nodes & rels) via APOC.
    """
    try:
        # 0) Create DB
        new_db = _create_new_database(explicit=dataset_name)

        # --------- EDGES ---------
        df_e = _read_csv(edges)
        src_col, dst_col, type_col = _detect_edges_columns(df_e)
        df_e = df_e.copy()
        if type_col and type_col in df_e.columns:
            df_e["_type"] = df_e[type_col].astype(str).apply(_sanitize_type)
        else:
            df_e["_type"] = "REL"

        # ----- timestamp handling (per your rules) -----
        # Case-insensitive match for provided timestamp column name
        tcol_resolved = None
        if isTemporal:
            if not timestampColumn or not str(timestampColumn).strip():
                raise HTTPException(
                    status_code=400,
                    detail="timestampColumn must be provided when isTemporal=true."
                )
            _lc_map = {c.lower(): c for c in df_e.columns}
            tcol_resolved = _lc_map.get(str(timestampColumn).strip().lower())
            if not tcol_resolved:
                raise HTTPException(
                    status_code=400,
                    detail=f"Timestamp column '{timestampColumn}' not found in edges CSV."
                )

            # Coerce to numeric; if any NaNs remain, fail fast with a clear error
            df_e["_timestamp"] = pd.to_numeric(df_e[tcol_resolved], errors="coerce")
            bad = int(df_e["_timestamp"].isna().sum())
            if bad > 0:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Timestamp column '{tcol_resolved}' contains {bad} non-numeric or missing value(s). "
                        "Please clean or choose a numeric/parseable column."
                    ),
                )
        else:
            # Non-temporal import: all edges get timestamp = 0.0
            df_e["_timestamp"] = 0.0


        ids = pd.unique(
            pd.concat([df_e[src_col], df_e[dst_col]], ignore_index=True).dropna()
        )

        _mk_constraint_entity(new_db)
        for chunk in _chunk(list(ids), batch_size):
            run(_with_db(new_db, _merge_nodes_query()), {"rows": _rows_for_nodes(chunk)})

        exclude = {src_col, dst_col}
        if type_col:
            exclude.add(type_col)
        exclude.update({"_type", "_timestamp"})
        prop_cols = [c for c in df_e.columns if c not in exclude]

        rel_total = 0

        def _rel_rows(sub: pd.DataFrame) -> List[Dict[str, Any]]:
            out = []
            for _, r in sub.iterrows():
                props = {k: r[k] for k in prop_cols if k in sub.columns}
                ts_val = r["_timestamp"]
                if pd.isna(ts_val):
                    ts_val = 0
                fv = float(ts_val)
                if fv.is_integer():
                    props["timestamp"] = int(fv)
                else:
                    props["timestamp"] = fv
                out.append({"src": r[src_col], "dst": r[dst_col], "props": props})
            return out

        for typ, grp in df_e.groupby("_type"):
            rows = _rel_rows(grp)
            for chunk in _chunk(rows, batch_size):
                run(_with_db(new_db, _merge_rels_query(typ)), {"rows": chunk})
                rel_total += len(chunk)

        # --------- FEATURES (optional, in-memory transform) ---------
        feat_total = 0
        feature_transform_summary: Optional[Dict[str, Any]] = None

        if features is not None:
            df_f = _read_csv(features)
            cols_lower = [c.lower() for c in df_f.columns]
            if "id" not in cols_lower:
                raise HTTPException(status_code=400, detail="Features CSV must have an 'id' column.")
            id_col = df_f.columns[cols_lower.index("id")]

            df_f_transformed, feature_transform_summary = _transform_feature_df(
                df_f, id_col=id_col, max_categories=max_categories
            )

            rows = _map_rows_for_props(df_f_transformed, id_col)
            feat_total = len(rows)
            for chunk in _chunk(rows, batch_size):
                run(_with_db(new_db, _set_props_query()), {"rows": chunk})

        # --------- OPTIONAL: PURGE ANY REMAINING STRING-ISH PROPS ---------
        purge_summary: Optional[Dict[str, Any]] = None
        if purge_all_stringish:
            purge_summary = _purge_stringish_props(new_db)

        return {
            "status": "ok",
            "database": new_db,
            "nodes_labels": {"Entity": len(ids)},
            "relationships_created_or_merged": rel_total,
            "feature_rows_processed": feat_total,
            "feature_transform_summary": feature_transform_summary or {
                "kept_numeric": [],
                "converted_one_hot_scalar": [],
                "dropped_string_with_nulls": [],
                "dropped_mixed": [],
                "skipped_too_many_categories": [],
            },
            "purged_stringish_properties": purge_summary or {
                "nodeKeysPurged": [],
                "relKeysPurged": [],
                "nodeAssignmentsDropped": 0,
                "relAssignmentsDropped": 0
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Load failed: {e}")
