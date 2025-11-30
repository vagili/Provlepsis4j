# app/routers/lp.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Tuple
import re
import math
import random
import numpy as np
import sys

from ..db import run, run_data, current_database

# Try to import sklearn; if missing, raise a helpful error.
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        accuracy_score,
        precision_recall_fscore_support,
    )
except Exception:
    LogisticRegression = None  # handled via _assert_sklearn()

router = APIRouter()

# ------------------------ DEBUG HELPER ------------------------

def _dbg(tag: str, **fields: Any) -> None:
    """
    Lightweight debug logger.
    Always tries to include the *current* database from current_database().
    """
    try:
        db = current_database()
    except Exception:
        db = "<unknown>"
    msg = " ".join(f"{k}={v!r}" for k, v in fields.items())
    print(f"[LP DEBUG] {tag} db={db} {msg}", file=sys.stderr, flush=True)


# ------------------------ GLOBAL STATE ------------------------

_ACTIVE_CACHE_DB: Optional[str] = None
_PREDICTED_EDGES: Dict[str, List[Tuple[str, str]]] = {}  # e.g. {"FastRP": [...], "Node2Vec": [...]}
# Canonical candidate non-edge pool (ordered) per active DB
_CANON_POOL: List[Tuple[str, str]] = []
_CANON_SIG: Tuple[int, int] = (0, 0)  # (pool_size, version)
_EMBED_KEYS = ("Node2Vec", "FastRP", "GraphSAGE", "HashGNN")
_POOL_BY_KEY: Dict[str, List[Tuple[str, str]]] = {}     # {"FastRP": [(s,t), ...], ...}
_POOL_VERSION: int = 1 
_METRICS_BY_KEY: Dict[str, Dict[str, Any]] = {}  
_TEST_PAIRS_BY_GRAPH: Dict[str, List[Tuple[str, str]]] = {}

_DEBUG_INITIAL_BATCH: Dict[str, List[Tuple[str, str]]] = {}

# ------------------------
# Request/Response models
# ------------------------

class LPBody(BaseModel):
    # Name of the node embedding property (e.g., "embedding_n2v_128", "embedding_fastrp_256")
    embeddingProperty: str = Field(..., description="Name of the node embedding property")
    # Graph names already created by your split
    trainGraphName: str = "trainGraph"
    valGraphName: Optional[str] = None
    testGraphName: Optional[str] = "testGraph"
    # Negatives per positive (1.0 => 1:1)
    negativeRatio: float = Field(default=1.0, ge=0.1, le=10.0)
    # Top-K prediction settings
    predictK: int = Field(default=100, ge=1)
    candidateMultiplier: int = Field(default=20, ge=2, le=200)
    # Probability threshold — keep only predicted edges with prob >= this
    probThreshold: float = Field(default=0.80, ge=0.0, le=1.0)
    # Output in-memory projection containing original + predicted edges
    outputGraphName: str = "predictedGraph"

    force_new_canonical_pool: bool = Field(
        False,
        description= ("If true, rebuild the negative candidate pool for this run"
        ),
    )

class LPMetrics(BaseModel):
    count_pos: int
    count_neg: int
    auc: Optional[float] = None
    ap: Optional[float] = None
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None

class LPResponse(BaseModel):
    ok: bool
    train: LPMetrics
    validation: Optional[LPMetrics] = None
    test: Optional[LPMetrics] = None
    model: Dict[str, Any]  # summary (algo, embedding prop, feature name)
    predicted: Optional[Dict[str, Any]] = None  # info about projected & materialized output

# ------------------------
# Helpers
# ------------------------

_PROP_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

def _set_debug_initial_batch(emb_key: str, pairs: List[Tuple[str, str]]) -> None:
    """
    Remember up to 100 initial-prediction edges for this embedding key.
    Only for debugging / comparison with iterative LP.
    """
    sample = pairs[:100]
    _DEBUG_INITIAL_BATCH[emb_key] = list(sample)
    _dbg(
        "_set_debug_initial_batch",
        emb_key=emb_key,
        count=len(sample),
        edges=sample,
    )


def _get_debug_initial_batch(emb_key: str) -> List[Tuple[str, str]]:
    """
    Retrieve the stored initial LP edges (up to 100) for this embedding key.
    Used by lp_iter.py to compare with its own batch.
    """
    batch = _DEBUG_INITIAL_BATCH.get(emb_key) or []
    _dbg(
        "_get_debug_initial_batch",
        emb_key=emb_key,
        count=len(batch),
    )
    return list(batch)


# Map the provided embeddingProperty to a canonical DB suffix
def _embedding_db_suffix(prop: str) -> str:
    s = (prop or "").lower()
    if "fastrp" in s:
        return "FastRP"
    if "node2vec" in s or "n2v" in s:
        return "Node2Vec"
    if "graphsage" in s or "sage" in s:
        return "GraphSAGE"
    if "hashgnn" in s or "hgnn" in s or ("hash" in s and "gnn" in s):
        return "HashGNN"
    # Fallback: sanitize the raw property into a readable suffix
    fallback = re.sub(r"[^A-Za-z0-9]+", "_", prop).strip("_")
    return fallback or "Embedding"

def _assert_sklearn():
    if LogisticRegression is None:
        _dbg("_assert_sklearn.missing")
        raise HTTPException(
            status_code=500,
            detail=(
                "scikit-learn is not installed in the backend image. "
                "Add it to requirements.txt (e.g., scikit-learn==1.4.*) and rebuild."
            ),
        )

def _sanitize_prop(name: str) -> str:
    if not _PROP_RE.match(name):
        _dbg("_sanitize_prop.invalid", name=name)
        raise HTTPException(status_code=400, detail=f"Invalid embedding property name: {name}")
    return name

def _stream_unique_pairs_from_gds(gname: str) -> List[Tuple[str, str]]:
    """
    Stream unique UNDIRECTED pairs (s < t by elementId) from an in-memory GDS graph.
    Returns [(s_eid, t_eid)] where s_eid, t_eid are elementId strings.
    """
    if not gname:
        _dbg("_stream_unique_pairs_from_gds.empty_name")
        return []
    _dbg("_stream_unique_pairs_from_gds.start", graph=gname)
    cypher = """
    CALL gds.graph.relationships.stream($g)
    YIELD sourceNodeId, targetNodeId
    WITH gds.util.asNode(sourceNodeId) AS sN, gds.util.asNode(targetNodeId) AS tN
    WITH elementId(sN) AS s, elementId(tN) AS t
    WITH CASE WHEN s < t THEN s ELSE t END AS s, CASE WHEN s < t THEN t ELSE s END AS t
    RETURN DISTINCT s, t
    """
    rows = run_data(cypher, {"g": gname})
    _dbg("_stream_unique_pairs_from_gds.done", graph=gname, pairs=len(rows))
    return [(r["s"], r["t"]) for r in rows]

def _sample_negatives(k: int) -> List[Tuple[str, str]]:
    """
    Sample ~k UNDIRECTED non-edges from the stored graph (global), using randomized elementId pairs;
    filter out existing edges of any type. Returns [(s_eid, t_eid)] with s < t.
    """
    if k <= 0:
        _dbg("_sample_negatives.skip", k=k)
        return []
    oversample = max(k * 3, 50)
    _dbg("_sample_negatives.start", k=k, oversample=oversample)
    cypher = """
    // (1) fetch ids and N
    MATCH (n)
    WITH collect(elementId(n)) AS ids, count(n) AS N

    // (2) sample candidate pairs
    CALL {
      WITH ids, N
      UNWIND range(1, $oversample) AS _
      WITH ids, N,
           ids[toInteger(rand() * N)] AS a,
           ids[toInteger(rand() * N)] AS b
      WITH a, b
      WHERE a < b
      RETURN DISTINCT a AS s, b AS t
      LIMIT $oversample
    }

    // (3) keep only pairs without an existing edge in the stored graph
    WITH s, t
    WHERE NOT EXISTS {
      MATCH (sN)--(tN)
      WHERE elementId(sN) = s AND elementId(tN) = t
    }
    RETURN DISTINCT s, t
    LIMIT $k
    """
    rows = run_data(cypher, {"k": k, "oversample": oversample})
    _dbg("_sample_negatives.done", returned=len(rows))
    return [(r["s"], r["t"]) for r in rows]

def _db_seed(dbname: str, version: int = 1) -> int:
    # Stable seed from DB name + small version to allow changing strategy later
    seed = abs(hash((dbname, "lp_pool_v1", version))) % (2**31 - 1)
    _dbg("_db_seed", dbname=dbname, version=version, seed=seed)
    return seed

def _existing_edge_pairs_limit(limit: int = 0) -> set[Tuple[str, str]]:
    """
    Optionally fetch existing edges (undirected) if you want to exclude them in Python.
    Not used if we filter inside Cypher; kept for reference.
    """
    _dbg("_existing_edge_pairs_limit.noop", limit=limit)
    return set()

def _build_canonical_candidate_pairs(pool_size: int, version: int = 1) -> List[Tuple[str, str]]:
    """
    Deterministically construct an ORDERED list of non-edge pairs (s<t by elementId)
    for the *current* base DB.

    Uses a PRNG seeded by (dbName, version) so that:
      - for a fixed DB + version, you always get the same pool
      - if DB changes, the pool changes automatically
    """
    base = _ensure_cache_for_current_db()
    seed = _db_seed(base, version)
    rng = random.Random(seed)
    _dbg("_build_canonical_candidate_pairs.start", base_db=base, pool_size=pool_size, version=version)

    # 1) get all node elementIds in this DB
    ids_rows = run_data("MATCH (n) RETURN elementId(n) AS eid ORDER BY eid", db=base)
    ids = [r["eid"] for r in ids_rows]
    N = len(ids)
    _dbg("_build_canonical_candidate_pairs.nodes", base_db=base, N=N)
    if N < 2:
        _dbg("_build_canonical_candidate_pairs.too_few_nodes", base_db=base)
        return []

    want = max(1, int(pool_size))
    target_raw = max(want * 3, want + 100)

    seen_idx: set[Tuple[int, int]] = set()
    raw_pairs: List[Tuple[str, str]] = []

    # 2) deterministically sample candidate pairs in Python
    while len(raw_pairs) < target_raw:
        i = rng.randrange(N)
        j = rng.randrange(N)
        if i == j:
            continue
        a, b = (i, j) if i < j else (j, i)
        key = (a, b)
        if key in seen_idx:
            continue
        seen_idx.add(key)
        raw_pairs.append((ids[a], ids[b]))

    _dbg("_build_canonical_candidate_pairs.raw_pairs", base_db=base, generated=len(raw_pairs))

    # 3) filter out existing edges in Cypher, keep order, cap to 'want'
    rows = run_data(
        """
        UNWIND $pairs AS P
        WITH P.s AS s, P.t AS t
        WHERE NOT EXISTS {
          MATCH (sN)--(tN)
          WHERE elementId(sN) = s AND elementId(tN) = t
        }
        RETURN s, t
        LIMIT $limit
        """,
        {"pairs": [{"s": s, "t": t} for (s, t) in raw_pairs], "limit": want},
        db=base,
    )
    _dbg("_build_canonical_candidate_pairs.filtered", base_db=base, kept=len(rows))
    return [(r["s"], r["t"]) for r in rows]


def _get_canonical_pool(
    min_needed: int,
    max_needed: int,
    version: int = 1,
) -> List[Tuple[str, str]]:
    """
    Get or build the canonical pool of negative candidate pairs for the *current* DB.

    - Single global pool per active DB (tracked by _ACTIVE_CACHE_DB)
    - Signature (max_needed, version) tells us whether reuse is valid
    """
    base = _ensure_cache_for_current_db()
    global _CANON_POOL, _CANON_SIG

    sig = (max_needed, version)
    _dbg(
        "_get_canonical_pool.enter",
        base_db=base,
        min_needed=min_needed,
        max_needed=max_needed,
        sig=sig,
        current_sig=_CANON_SIG,
        current_len=len(_CANON_POOL),
    )

    if _CANON_POOL and _CANON_SIG == sig and len(_CANON_POOL) >= min_needed:
        _dbg("_get_canonical_pool.cache_hit", base_db=base, pool_len=len(_CANON_POOL))
        return _CANON_POOL

    _dbg("_get_canonical_pool.rebuild", base_db=base, old_sig=_CANON_SIG, new_sig=sig)
    pool = _build_canonical_candidate_pairs(pool_size=max_needed, version=version)
    _CANON_POOL = pool
    _CANON_SIG = sig
    _dbg("_get_canonical_pool.built", base_db=base, pool_len=len(pool), sig=_CANON_SIG)
    return pool


def _ensure_pool_for_key(emb_key: str, ensure_remaining: int, version: int = 1) -> List[Tuple[str, str]]:
    """
    Ensure the per-algorithm pool has at least `ensure_remaining` *remaining* candidates.
    Uses the canonical pool for the current DB and version.
    """
    base = _ensure_cache_for_current_db()
    _dbg("_ensure_pool_for_key.enter", base_db=base, emb_key=emb_key, ensure_remaining=ensure_remaining)

    max_needed = max(ensure_remaining * 3, ensure_remaining + 200)
    canon = _get_canonical_pool(min_needed=ensure_remaining, max_needed=max_needed, version=version)

    cur = _POOL_BY_KEY.get(emb_key) or []
    if not cur:
        _POOL_BY_KEY[emb_key] = list(canon)
        _dbg("_ensure_pool_for_key.init", base_db=base, emb_key=emb_key, pool_len=len(_POOL_BY_KEY[emb_key]))
        return _POOL_BY_KEY[emb_key]

    have = set(cur)
    appended = False
    for p in canon:
        if p not in have:
            cur.append(p)
            appended = True

    if appended:
        _POOL_BY_KEY[emb_key] = cur
        _dbg("_ensure_pool_for_key.extend", base_db=base, emb_key=emb_key, pool_len=len(cur))

    return cur

def _ensure_cache_for_current_db() -> str:
    """
    Ensure our canonical-pool cache is aligned to the *current* Neo4j database.

    If the user switches DB (via your /db/select), we:
      - reset the canonical pool
      - reset per-embedding pools
      - reset in-memory predicted edges
    and remember the new base in _ACTIVE_CACHE_DB.

    Returns the current base DB name.
    """
    global _ACTIVE_CACHE_DB, _CANON_POOL, _CANON_SIG, _POOL_BY_KEY, _PREDICTED_EDGES, _METRICS_BY_KEY, _TEST_PAIRS_BY_GRAPH

    db = current_database()
    if db != _ACTIVE_CACHE_DB:
        _dbg("_ensure_cache_for_current_db.switch", old=_ACTIVE_CACHE_DB, new=db)
        _ACTIVE_CACHE_DB = db
        _CANON_POOL = []
        _CANON_SIG = (0, 0)
        _POOL_BY_KEY = {}
        _PREDICTED_EDGES.clear()
        _METRICS_BY_KEY = {}
        _TEST_PAIRS_BY_GRAPH = {}
    return db


def _predict_from_per_alg_pool(
    emb_key: str,
    emb: Dict[str, np.ndarray],
    model: LogisticRegression,
    k: int,
    threshold: float,
    candidate_multiplier: int,
    avoid_set: Optional[set[Tuple[str, str]]] = None,
) -> List[Tuple[str, str, float]]:
    """
    Use/extend the per-algorithm non-edge pool; deterministic across runs.
    """
    want = max(1, int(k))
    thr = float(threshold)
    avoid = set(avoid_set or set())
    _dbg(
        "_predict_from_per_alg_pool.enter",
        emb_key=emb_key,
        want=want,
        thr=thr,
        candidate_multiplier=candidate_multiplier,
        avoid_size=len(avoid),
    )

    remaining_target = max(want * candidate_multiplier, want + 50)
    pool = _ensure_pool_for_key(emb_key, ensure_remaining=remaining_target, version=_POOL_VERSION)
    _dbg("_predict_from_per_alg_pool.pool", emb_key=emb_key, pool_len=len(pool))

    scored_keep: List[Tuple[str, str, float]] = []
    scored_buffer: List[Tuple[str, str, float]] = []

    for s, t in pool:
        key = _pair_key(s, t)
        if key in avoid:
            continue
        es = emb.get(s); et = emb.get(t)
        if es is None or et is None:
            continue
        p = float(model.predict_proba((es * et).reshape(1, -1))[0, 1])
        if p >= thr:
            scored_keep.append((key[0], key[1], p))
            if len(scored_keep) >= want:
                break
        else:
            scored_buffer.append((key[0], key[1], p))

    _dbg("_predict_from_per_alg_pool.after_scan", emb_key=emb_key,
         keep=len(scored_keep), buffer=len(scored_buffer))

    if len(scored_keep) < want:
        if len(scored_buffer) < want:
            _ensure_pool_for_key(emb_key, ensure_remaining=remaining_target * 2, version=_POOL_VERSION)
        scored_buffer.sort(key=lambda x: x[2], reverse=True)
        need = want - len(scored_keep)
        scored_keep.extend(scored_buffer[:need])

    if scored_keep:
        keep_set = {_pair_key(s, t) for (s, t, _p) in scored_keep}
        before_len = len(_POOL_BY_KEY[emb_key])
        _POOL_BY_KEY[emb_key] = [
            pair for pair in _POOL_BY_KEY[emb_key]
            if _pair_key(*pair) not in keep_set
        ]
        after_len = len(_POOL_BY_KEY[emb_key])
        _dbg("_predict_from_per_alg_pool.trim_pool", emb_key=emb_key,
             removed=before_len - after_len, new_pool_len=after_len)

    _dbg("_predict_from_per_alg_pool.done", emb_key=emb_key, kept=len(scored_keep))
    return scored_keep

def _predict_with_test_priority(
    emb_key: str,
    emb: Dict[str, np.ndarray],
    model: LogisticRegression,
    k: int,
    threshold: float,
    candidate_multiplier: int,
    test_pairs: Optional[list[tuple[str, str]]] = None,
    avoid_set: Optional[set[tuple[str, str]]] = None,
) -> list[tuple[str, str, float]]:
    """
    First scan the (ordered) test-set edges and keep those whose prob >= threshold.
    If we still have fewer than k, fill the remainder from the canonical negative
    pool using _predict_from_per_alg_pool.

    - test_pairs MUST be canonicalised (s < t) for determinism.
    - avoid_set is a set of canonical pairs (s, t) already predicted for this emb_key.
    """
    want = max(1, int(k))
    thr = float(threshold)
    avoid = set(avoid_set or set())

    _dbg(
        "_predict_with_test_priority.enter",
        emb_key=emb_key,
        want=want,
        thr=thr,
        test_pairs=len(test_pairs or []),
        avoid_size=len(avoid),
        candidate_multiplier=candidate_multiplier,
    )

    kept: list[tuple[str, str, float]] = []
    seen_test: set[tuple[str, str]] = set()

    # --- Stage 1: scan test-set edges in fixed order ---
    for s, t in (test_pairs or []):
        key = _pair_key(s, t)
        if key in avoid or key in seen_test:
            continue

        es = emb.get(key[0])
        et = emb.get(key[1])
        if es is None or et is None:
            continue

        p = float(model.predict_proba((es * et).reshape(1, -1))[0, 1])
        if p >= thr:
            kept.append((key[0], key[1], p))
            seen_test.add(key)
            if len(kept) >= want:
                _dbg("_predict_with_test_priority.test_done", emb_key=emb_key, kept=len(kept))
                return kept  # got K purely from test edges

    _dbg("_predict_with_test_priority.after_test", emb_key=emb_key, kept=len(kept))

    # --- Stage 2: fill remainder from canonical negative pool ---
    if len(kept) < want:
        extra_avoid = avoid | {_pair_key(s, t) for (s, t, _p) in kept}
        need = want - len(kept)
        more = _predict_from_per_alg_pool(
            emb_key=emb_key,
            emb=emb,
            model=model,
            k=need,
            threshold=threshold,
            candidate_multiplier=candidate_multiplier,
            avoid_set=extra_avoid,
        )
        kept.extend(more)

    _dbg("_predict_with_test_priority.done", emb_key=emb_key, kept=len(kept))
    return kept


def _load_embeddings(prop: str) -> Dict[str, np.ndarray]:
    """
    Read elementId -> embedding (numpy array) for all nodes where property exists.
    """
    prop = _sanitize_prop(prop)
    _dbg("_load_embeddings.start", prop=prop)
    cypher = f"""
    MATCH (n)
    WHERE n.{prop} IS NOT NULL
    RETURN elementId(n) AS eid, n.{prop} AS emb
    """
    rows = run_data(cypher)
    embs: Dict[str, np.ndarray] = {}
    for r in rows:
        v = r["emb"]
        if isinstance(v, list):
            embs[r["eid"]] = np.array(v, dtype=float)
    _dbg("_load_embeddings.done", prop=prop, count=len(embs))
    return embs

def _hadamard(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a * b

def _build_set_features(
    pos_pairs: List[Tuple[str, str]],
    emb: Dict[str, np.ndarray],
    negative_ratio: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Build X (features) and y (labels) for a set.
    """
    _dbg("_build_set_features.start", pos_pairs=len(pos_pairs), negative_ratio=negative_ratio)
    X_pos: List[np.ndarray] = []
    for s, t in pos_pairs:
        es = emb.get(s)
        et = emb.get(t)
        if es is None or et is None:
            continue
        X_pos.append(_hadamard(es, et))
    cnt_pos = len(X_pos)

    want_neg = int(math.ceil(cnt_pos * max(0.0, min(10.0, negative_ratio))))
    neg_pairs = _sample_negatives(want_neg)
    X_neg: List[np.ndarray] = []
    for s, t in neg_pairs:
        es = emb.get(s)
        et = emb.get(t)
        if es is None or et is None:
            continue
        X_neg.append(_hadamard(es, et))
    cnt_neg = len(X_neg)

    if X_pos or X_neg:
        X = np.vstack([X_pos, X_neg]) if X_neg else np.vstack(X_pos)
        y = np.hstack([np.ones(cnt_pos, dtype=int), np.zeros(cnt_neg, dtype=int)]) if X_neg else np.ones(cnt_pos, dtype=int)
    else:
        X = np.zeros((0, 1))
        y = np.zeros((0,), dtype=int)

    _dbg("_build_set_features.done", cnt_pos=cnt_pos, cnt_neg=cnt_neg, X_shape=X.shape)
    return X, y, cnt_pos, cnt_neg

def _eval_scores(y_true: np.ndarray, y_prob: np.ndarray) -> LPMetrics:
    y_hat = (y_prob >= 0.5).astype(int)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = None
    try:
        ap = average_precision_score(y_true, y_prob)
    except Exception:
        ap = None
    try:
        acc = accuracy_score(y_true, y_hat)
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_hat, average="binary", zero_division=0
        )
    except Exception:
        acc = p = r = f1 = None

    met = LPMetrics(
        count_pos=int((y_true == 1).sum()),
        count_neg=int((y_true == 0).sum()),
        auc=auc, ap=ap, accuracy=acc, precision=p, recall=r, f1=f1,
    )
    _dbg(
        "_eval_scores",
        count_pos=met.count_pos,
        count_neg=met.count_neg,
        auc=met.auc,
        ap=met.ap,
        acc=met.accuracy,
        precision=met.precision,
        recall=met.recall,
        f1=met.f1,
    )
    return met

def _predict_topk_edges(
    emb: Dict[str, np.ndarray],
    model: LogisticRegression,
    k: int,
    candidate_multiplier: int,
) -> List[Tuple[str, str, float]]:
    """
    Legacy sampler (not used with canonical pool); kept for reference.
    """
    pool = max(k * candidate_multiplier, k + 5)
    _dbg("_predict_topk_edges", k=k, pool=pool)
    cand = _sample_negatives(pool)
    scored: List[Tuple[str, str, float]] = []
    for s, t in cand:
        es = emb.get(s)
        et = emb.get(t)
        if es is None or et is None:
            continue
        x = _hadamard(es, et).reshape(1, -1)
        p = float(model.predict_proba(x)[0, 1])
        scored.append((s, t, p))
    scored.sort(key=lambda x: x[2], reverse=True)
    _dbg("_predict_topk_edges.done", returned=len(scored[:k]))
    return scored[:k]


def _predict_at_least_k_above_threshold(
    emb: Dict[str, np.ndarray],
    model: LogisticRegression,
    k: int,
    threshold: float,
    start_factor: int = 20,
    max_factor: int = 200,
    growth: float = 2.0,
    max_rounds: int = 5,
    avoid_set: Optional[set[tuple[str, str]]] = None,  # NEW
) -> List[Tuple[str, str, float]]:
    """
    Deterministic pool. Return exactly K *new* pairs (s,t,p) such that
    (s,t) ∉ avoid_set (undirected). If not enough pass the threshold,
    keep scanning the pool; if still short, fallback to best unseen.
    """
    want = max(1, int(k))
    thr = float(threshold)
    avoid = set(avoid_set or set())
    _dbg(
        "_predict_at_least_k_above_threshold.enter",
        want=want,
        thr=thr,
        start_factor=start_factor,
        max_factor=max_factor,
        avoid_size=len(avoid),
    )

    max_needed = min(max(want * max(start_factor, 1), want + 5), want * max_factor)
    pool = _get_canonical_pool(min_needed=want, max_needed=max_needed, version=1)
    _dbg("_predict_at_least_k_above_threshold.pool", pool_len=len(pool))

    kept: List[Tuple[str, str, float]] = []
    seen_batch: set[tuple[str, str]] = set()

    def add_if_new(s: str, t: str, p: float) -> bool:
        key = _pair_key(s, t)
        if key in avoid or key in seen_batch:
            return False
        kept.append((key[0], key[1], p))
        seen_batch.add(key)
        return True

    for s, t in pool:
        es = emb.get(s); et = emb.get(t)
        if es is None or et is None:
            continue
        p = float(model.predict_proba((es * et).reshape(1, -1))[0, 1])
        if p >= thr and add_if_new(s, t, p):
            if len(kept) >= want:
                _dbg("_predict_at_least_k_above_threshold.threshold_done", kept=len(kept))
                return kept

    if len(kept) < want:
        scored: List[Tuple[str, str, float]] = []
        for s, t in pool:
            key = _pair_key(s, t)
            if key in avoid or key in seen_batch:
                continue
            es = emb.get(s); et = emb.get(t)
            if es is None or et is None:
                continue
            p = float(model.predict_proba((es * et).reshape(1, -1))[0, 1])
            scored.append((key[0], key[1], p))
        scored.sort(key=lambda x: x[2], reverse=True)
        needed = want - len(kept)
        kept.extend(scored[:needed])

    _dbg("_predict_at_least_k_above_threshold.done", kept=len(kept))
    return kept  # may be < K if graph is too dense/small


def _project_predicted_graph(graph_name: str, predicted_triples: list[tuple[str, str, float]]):
    """
    In-memory graph with:
      • all stored undirected edges (both directions)
      • predicted undirected edges (both directions, type 'PREDICTED', properties {probability})
    """
    _dbg("_project_predicted_graph.start", graph_name=graph_name, predicted=len(predicted_triples))
    run("CALL gds.graph.drop($name, false)", {"name": graph_name})

    node_q = "MATCH (n) RETURN id(n) AS id"

    rel_q = """
    // (A) All stored edges → canonicalize by *numeric* id(), then emit both directions
    MATCH (a)-[r]-(b)
    WITH id(a) AS aId, id(b) AS bId, type(r) AS typ
    WITH
      CASE WHEN aId < bId THEN aId ELSE bId END AS s,
      CASE WHEN aId < bId THEN bId ELSE aId END AS t,
      typ
    WITH collect(DISTINCT [s,t,typ]) AS pairs
    UNWIND pairs AS p
    UNWIND [[p[0],p[1]],[p[1],p[0]]] AS dir
    RETURN dir[0] AS source, dir[1] AS target, p[2] AS type, {} AS properties

    UNION

    // (B) Predicted pairs are given as elementId strings → resolve to numeric id(), emit both directions
    UNWIND $predicted AS P
    MATCH (sNode) WHERE elementId(sNode) = P.s
    MATCH (tNode) WHERE elementId(tNode) = P.t
    WITH id(sNode) AS s, id(tNode) AS t, toFloat(P.p) AS p
    UNWIND [[s,t],[t,s]] AS dir
    RETURN dir[0] AS source, dir[1] AS target, 'PREDICTED' AS type, {probability: p} AS properties
    """

    config = {
        "parameters": {
            "predicted": [{"s": s, "t": t, "p": float(p)} for (s, t, p) in predicted_triples]
        }
    }

    rows = run_data(
        """
        CALL gds.graph.project.cypher($name, $nodeQuery, $relQuery, $config)
        YIELD graphName, nodeCount, relationshipCount
        RETURN graphName, nodeCount, relationshipCount
        """,
        {
            "name": graph_name,
            "nodeQuery": node_q,
            "relQuery": rel_q,
            "config": config,
        },
    )
    if rows:
        _dbg("_project_predicted_graph.done",
             graphName=rows[0]["graphName"],
             nodeCount=rows[0]["nodeCount"],
             relationshipCount=rows[0]["relationshipCount"])
    else:
        _dbg("_project_predicted_graph.empty_result")
    return rows[0] if rows else {"graphName": graph_name, "nodeCount": 0, "relationshipCount": 0}

# After: _EMBED_KEYS = ("Node2Vec", "FastRP", "GraphSAGE", "HashGNN")

def _get_emb_key_from_prop(prop: str) -> str:
    return _embedding_db_suffix(prop)

def _get_predicted_set_for_key(emb_key: str) -> set[tuple[str, str]]:
    lst = _PREDICTED_EDGES.get(emb_key) or []
    s = {_pair_key(s, t) for (s, t) in lst}
    _dbg("_get_predicted_set_for_key", emb_key=emb_key, size=len(s))
    return s

def _append_memory_for_key(emb_key: str, new_pairs: list[tuple[str, str]]) -> None:
    prev = _PREDICTED_EDGES.get(emb_key) or []
    prev_set = {_pair_key(s, t) for (s, t) in prev}
    added = 0
    for s, t in new_pairs:
        k = _pair_key(s, t)
        if k not in prev_set:
            prev.append((k[0], k[1]))
            prev_set.add(k)
            added += 1
    _PREDICTED_EDGES[emb_key] = prev
    _dbg("_append_memory_for_key", emb_key=emb_key, added=added, total=len(prev))


def _materialize_prediction_db(
    gname: str = "predictedGraph",
    node_key: str = "eid",
    embedding_suffix: Optional[str] = None,
    predicted_triples: Optional[list[tuple[str, str, float]]] = None,
) -> str:
    pred_list = [{"s": s, "t": t, "p": float(p)} for (s, t, p) in (predicted_triples or [])]
    base_db = current_database()
    suffix = f"-predicted-{embedding_suffix}" if embedding_suffix else "-predicted"
    target_db = f"{base_db}{suffix}"
    _dbg("_materialize_prediction_db.start", gname=gname, base_db=base_db, target_db=target_db,
         predicted=len(pred_list))

    rows_ts = run_data(
        """
        MATCH ()-[r]-()
        WITH CASE WHEN r.timestamp IS NOT NULL THEN r.timestamp ELSE 0 END AS ts
        RETURN coalesce(max(ts), 0) AS max_ts
        """,
        db=base_db,
    )
    max_ts = rows_ts[0]["max_ts"] if rows_ts else 0
    max_pred_ts = max_ts + 1
    _dbg("_materialize_prediction_db.max_ts", max_ts=max_ts, max_pred_ts=max_pred_ts)

    _recreate_database(target_db)

    run(
        f"CREATE CONSTRAINT node_{node_key}_uniq IF NOT EXISTS "
        f"FOR (n:Entity) REQUIRE n.{node_key} IS UNIQUE",
        db=target_db,
    )

    node_rows = run_data(
        """
        CALL gds.graph.relationships.stream($g)
        YIELD sourceNodeId, targetNodeId
        WITH gds.util.asNode(sourceNodeId) AS sN,
             gds.util.asNode(targetNodeId) AS tN
        WITH collect(sN) + collect(tN) AS nodes
        UNWIND nodes AS n
        WITH DISTINCT n
        RETURN elementId(n)      AS eid,
               labels(n)         AS labels,
               properties(n)     AS props
        """,
        {"g": gname},
    )
    _dbg("_materialize_prediction_db.nodes", target_db=target_db, count=len(node_rows))

    if node_rows:
        run(
            """
            UNWIND $batch AS row
            CALL apoc.merge.node(row.labels, {eid: row.eid}, row.props, row.props) YIELD node
            RETURN count(*) AS upserted
            """,
            {"batch": node_rows},
            db=target_db,
        )

    rel_rows = run_data(
        """
        CALL gds.graph.relationships.stream($g)
        YIELD sourceNodeId, targetNodeId, relationshipType
        WITH gds.util.asNode(sourceNodeId) AS sN,
            gds.util.asNode(targetNodeId) AS tN,
            relationshipType AS type
        WITH elementId(sN) AS a, elementId(tN) AS b, type
        WITH CASE WHEN a < b THEN a ELSE b END AS sid,
            CASE WHEN a < b THEN b ELSE a END AS tid,
            type

        // find timestamp of original edge, if it exists in the base DB
        CALL {
        WITH sid, tid, type
        OPTIONAL MATCH (s)-[r]-(t)
        WHERE elementId(s) = sid AND elementId(t) = tid AND type(r) = type
        RETURN coalesce(r.timestamp, 0.0) AS ts
        }

        // compute probability from the passed-in predictions for PREDICTED edges
        WITH sid, tid, type, ts,
            [p IN $pred WHERE p.s = sid AND p.t = tid | toFloat(p.p)] AS probs
        WITH sid, tid, type, ts,
            (CASE WHEN size(probs) > 0 THEN probs[0] ELSE 0.0 END) AS probability

        WITH DISTINCT sid, tid, type,
            CASE WHEN type = 'PREDICTED' THEN 1 ELSE 0 END AS predicted,
            CASE WHEN type = 'PREDICTED' THEN $maxPredTs ELSE ts END AS timestamp,
            probability
        RETURN sid, tid, type, predicted, timestamp, probability
        """,
        {"g": gname, "maxPredTs": max_pred_ts, "pred": pred_list},
    )
    _dbg("_materialize_prediction_db.rels", target_db=target_db, count=len(rel_rows))

    if rel_rows:
        run(
            f"""
            UNWIND $batch AS row
            MATCH (s {{ {node_key}: row.sid }}),
                (t {{ {node_key}: row.tid }})
            CALL apoc.merge.relationship(
            s,
            row.type,
            {{}},
            {{predicted: row.predicted, timestamp: row.timestamp, probability: row.probability}},
            t
            ) YIELD rel
            RETURN count(*) AS upserted
            """,
            {"batch": rel_rows},
            db=target_db,
        )

    _dbg("_materialize_prediction_db.done", target_db=target_db)
    return target_db

def _recreate_database(dbname: str, timeout_s: float = 30.0) -> None:
    _dbg("_recreate_database.start", dbname=dbname, timeout_s=timeout_s)
    run("DROP DATABASE $db IF EXISTS", {"db": dbname}, db="system")
    run("CREATE DATABASE $db", {"db": dbname}, db="system")

    import time
    start = time.time()
    while True:
        rows = run_data(
            "SHOW DATABASE $db YIELD name, currentStatus RETURN currentStatus",
            {"db": dbname},
            db="system",
        )
        status = (rows[0]["currentStatus"].lower() if rows else None)
        if status == "online":
            _dbg("_recreate_database.online", dbname=dbname, elapsed=time.time() - start)
            break
        if time.time() - start > timeout_s:
            _dbg("_recreate_database.timeout", dbname=dbname, status=status)
            raise HTTPException(
                status_code=503,
                detail=f"Database '{dbname}' did not come ONLINE (status={status})"
            )
        time.sleep(0.5)

def _gds_graph_exists(name: str) -> bool:
    try:
        rows = run_data("CALL gds.graph.exists($name) YIELD exists RETURN exists", {"name": name})
        exists = bool(rows and rows[0]["exists"])
        _dbg("_gds_graph_exists.exists", name=name, exists=exists)
        return exists
    except Exception:
        rows = run_data("CALL gds.graph.list($name) YIELD graphName RETURN graphName", {"name": name})
        exists = bool(rows)
        _dbg("_gds_graph_exists.list", name=name, exists=exists)
        return exists

# ---- helpers for heatmap (place near _gds_graph_exists) ----
def _pair_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)

def _get_test_pairs(graph_name: str = "testGraph") -> list[tuple[str, str]]:
    """
    Return a canonical, ORDERED list of test-set edges (s, t) with s < t
    for the given GDS graph, cached per base DB + graph name.
    """
    base = _ensure_cache_for_current_db()
    global _TEST_PAIRS_BY_GRAPH

    cached = _TEST_PAIRS_BY_GRAPH.get(graph_name)
    if cached is not None:
        _dbg("_get_test_pairs.cached_hit", base_db=base, graph=graph_name, count=len(cached))
        return cached

    if not _gds_graph_exists(graph_name):
        # IMPORTANT: do *not* cache an empty result here.
        # That way, once the split creates testGraph, a later call
        # will rebuild the pairs instead of being stuck at [].
        _dbg("_get_test_pairs.no_graph", base_db=base, graph=graph_name)
        return []

    # Use the existing streamer, then canonicalise + sort for deterministic order
    raw = _stream_unique_pairs_from_gds(graph_name)
    canon = sorted({_pair_key(s, t) for (s, t) in raw})
    _TEST_PAIRS_BY_GRAPH[graph_name] = canon
    _dbg("_get_test_pairs.built", base_db=base, graph=graph_name, count=len(canon))
    return canon


def _load_predicted_pairs_from_db(dbname: str) -> list[tuple[str, str]]:
    """
    Read undirected predicted pairs from a *materialized* predicted database by name.
    """
    _dbg("_load_predicted_pairs_from_db.start", dbname=dbname)
    try:
        rows = run_data(
            """
            MATCH (x)-[:PREDICTED]-(y)
            WITH elementId(x) AS a, elementId(y) AS b
            WITH CASE WHEN a < b THEN a ELSE b END AS s,
                 CASE WHEN a < b THEN b ELSE a END AS t
            RETURN DISTINCT s, t
            """,
            db=dbname,
        )
        _dbg("_load_predicted_pairs_from_db.done", dbname=dbname, count=len(rows))
        return [(r["s"], r["t"]) for r in rows]
    except Exception as e:
        _dbg("_load_predicted_pairs_from_db.error", dbname=dbname, error=str(e))
        return []

# ------------------------
# Core logic
# ------------------------

def _run_lp_logic(body: LPBody) -> Dict[str, Any]:

    base_db = _ensure_cache_for_current_db()

    if body.force_new_canonical_pool:
        global _CANON_POOL, _CANON_SIG, _POOL_BY_KEY
        _dbg("_run_lp_logic.force_new_canonical_pool", base_db=base_db)
        _CANON_POOL = []
        _CANON_SIG = (0, 0)
        _POOL_BY_KEY = {}


    _dbg("_run_lp_logic.enter", body=body.dict())
    _assert_sklearn()
    prop = _sanitize_prop(body.embeddingProperty)

    emb = _load_embeddings(prop)
    if not emb:
        _dbg("_run_lp_logic.no_embeddings", prop=prop)
        raise HTTPException(
            status_code=400,
            detail=f"No nodes found with embedding property '{prop}'."
        )

    pos_train = _stream_unique_pairs_from_gds(body.trainGraphName)
    pos_val = []
    if body.valGraphName and _gds_graph_exists(body.valGraphName):
        pos_val = _stream_unique_pairs_from_gds(body.valGraphName)
    pos_test = []
    if body.testGraphName and _gds_graph_exists(body.testGraphName):
        #pos_test = _stream_unique_pairs_from_gds(body.testGraphName)
        pos_test = _get_test_pairs(body.testGraphName)


    _dbg("_run_lp_logic.pos_pairs",
         train=len(pos_train),
         val=len(pos_val),
         test=len(pos_test))

    Xtr, ytr, npos_tr, nneg_tr = _build_set_features(pos_train, emb, body.negativeRatio)
    if Xtr.shape[0] == 0:
        _dbg("_run_lp_logic.no_training_pairs")
        raise HTTPException(status_code=400, detail="No usable training pairs (check embeddings exist for endpoints).")

    model = LogisticRegression(max_iter=200, n_jobs=None)
    model.fit(Xtr, ytr)
    prob_tr = model.predict_proba(Xtr)[:, 1]
    mtr = _eval_scores(ytr, prob_tr)
    mtr.count_pos = npos_tr
    mtr.count_neg = nneg_tr

    mval = None
    if pos_val:
        Xv, yv, npos_v, nneg_v = _build_set_features(pos_val, emb, body.negativeRatio)
        if Xv.shape[0] > 0:
            prob_v = model.predict_proba(Xv)[:, 1]
            mval = _eval_scores(yv, prob_v)
            mval.count_pos = npos_v
            mval.count_neg = nneg_v

    mts = None
    if pos_test:
        Xte, yte, npos_te, nneg_te = _build_set_features(pos_test, emb, body.negativeRatio)
        if Xte.shape[0] > 0:
            prob_te = model.predict_proba(Xte)[:, 1]
            mts = _eval_scores(yte, prob_te)
            mts.count_pos = npos_te
            mts.count_neg = nneg_te

    # ---remember metrics for this embedding family in this DB ---
    emb_key = _get_emb_key_from_prop(prop)   # e.g. "FastRP", "Node2Vec", ...
    base_db = _ensure_cache_for_current_db()
    _METRICS_BY_KEY[emb_key] = {
        "train": mtr.model_dump(),
        "validation": mval.model_dump() if mval else None,
        "test": mts.model_dump() if mts else None,
    }

    # ------------------------
    # Prediction phase
    # ------------------------
    emb_key = _get_emb_key_from_prop(prop)
    avoid_set_initial = _get_predicted_set_for_key(emb_key)
    _dbg(
        "_run_lp_logic.predict_phase",
        emb_key=emb_key,
        avoid_size=len(avoid_set_initial),
        predictK=body.predictK,
        probThreshold=body.probThreshold,
        candidateMultiplier=body.candidateMultiplier,
        test_pairs=len(pos_test),
    )

    # ---- A) DIAGNOSTIC / HEATMAP PREDICTIONS ----
    # Use test-set priority to create a "diagnostic" prediction set.
    # This is used ONLY for:
    #   - heatmap overlaps
    #   - the in-memory _PREDICTED_EDGES cache
    diag_filtered = _predict_with_test_priority(
        emb_key=emb_key,
        emb=emb,
        model=model,
        k=body.predictK,
        threshold=body.probThreshold,
        candidate_multiplier=body.candidateMultiplier,
        test_pairs=pos_test,
        avoid_set=avoid_set_initial,
    )

    diag_triples = [(s, t, p) for (s, t, p) in diag_filtered]
    diag_pairs   = [(s, t)     for (s, t, _p) in diag_filtered]
    _dbg(
        "_run_lp_logic.diag_predicted",
        emb_key=emb_key,
        triples=len(diag_triples),
        pairs=len(diag_pairs),
    )

    # Expose the initial diagnostic batch for lp_iter debugging
    _set_debug_initial_batch(emb_key, diag_pairs)

    # Remember these in memory so:
    #   - /lp/predicted/edges
    #   - heatmap
    #   - iterative LP
    # all see them.
    _append_memory_for_key(emb_key, diag_pairs)

    # ---- B) MATERIALISATION PREDICTIONS (K NEW EDGES) ----
    # Build a set of all existing ground-truth pairs (train/val/test),
    # canonicalised as (s<t) using _pair_key.
    gt_pairs = {
        _pair_key(s, t)
        for (s, t) in pos_train
    } | {
        _pair_key(s, t)
        for (s, t) in pos_val
    } | {
        _pair_key(s, t)
        for (s, t) in pos_test
    }

    # Avoid any pair that:
    #   - has already been predicted in this DB / embedding family, OR
    #   - is ground truth (train/val/test).
    avoid_for_db = _get_predicted_set_for_key(emb_key) | gt_pairs

    _dbg(
        "_run_lp_logic.materialize_phase",
        emb_key=emb_key,
        avoid_for_db_size=len(avoid_for_db),
        gt_pairs=len(gt_pairs),
    )

    # Now ask the per-algorithm pool for K *new* edges, not in avoid_for_db.
    new_triples_for_db = _predict_from_per_alg_pool(
        emb_key=emb_key,
        emb=emb,
        model=model,
        k=body.predictK,
        threshold=body.probThreshold,
        candidate_multiplier=body.candidateMultiplier,
        avoid_set=avoid_for_db,
    )
    new_pairs_for_db = [(s, t) for (s, t, _p) in new_triples_for_db]

    _dbg(
        "_run_lp_logic.materialize_new",
        emb_key=emb_key,
        triples=len(new_triples_for_db),
        pairs=len(new_pairs_for_db),
    )

    # Also record these truly-new edges into memory so the heatmap / iteration
    # see them as part of the "already predicted" pool.
    _append_memory_for_key(emb_key, new_pairs_for_db)

    # ---- C) PROJECT & MATERIALISE USING ONLY THE NEW EDGES ----
    proj_info = _project_predicted_graph(body.outputGraphName, new_triples_for_db)
    proj_info["keptAboveThreshold"] = len(new_pairs_for_db)
    proj_info["threshold"] = float(body.probThreshold)

    # expose diagnostic vs materialised sizes for debugging / UI
    proj_info["diagnosticPredictions"] = {
        "pairs": len(diag_pairs),
        "triples": len(diag_triples),
    }
    proj_info["materializedPredictions"] = {
        "pairs": len(new_pairs_for_db),
        "triples": len(new_triples_for_db),
    }

    base_db = _ensure_cache_for_current_db()
    emb_key = _embedding_db_suffix(prop)
    proj_info["cache"] = {
        "db": base_db,
        "currentKey": emb_key,
        "counts": {k: len(_PREDICTED_EDGES.get(k, [])) for k in _EMBED_KEYS},
        "edgesForCurrentKey": _PREDICTED_EDGES.get(emb_key, []),
    }
    _dbg("_run_lp_logic.cache_state", cache=proj_info["cache"])

    suffix = _embedding_db_suffix(prop)
    target_db = _materialize_prediction_db(
        gname=body.outputGraphName,
        embedding_suffix=suffix,
        predicted_triples=new_triples_for_db,
    )
    proj_info["materializedDatabase"] = target_db
    _dbg("_run_lp_logic.materialized", materialized_db=target_db)


    return {
        "ok": True,
        "train": mtr.model_dump(),
        "validation": mval.model_dump() if mval else None,
        "test": mts.model_dump() if mts else None,
        "model": {"algorithm": "LogisticRegression", "embeddingProperty": prop, "features": "Hadamard"},
        "predicted": proj_info,
    }

# --- helpers to list predicted variants for the current DB ---
def _db_exists(dbname: str) -> bool:
    try:
        rows = run_data(
            "SHOW DATABASE $db YIELD name, currentStatus RETURN currentStatus",
            {"db": dbname},
            db="system",
        )
        exists = bool(rows)
        _dbg("_db_exists", dbname=dbname, exists=exists)
        return exists
    except Exception as e:
        _dbg("_db_exists.error", dbname=dbname, error=str(e))
        return False

def _predicted_candidates_for(base_db: str) -> list[dict]:
    keys = ("Node2Vec", "FastRP", "GraphSAGE", "HashGNN")
    out = []
    for k in keys:
        name = f"{base_db}-predicted-{k}"
        exists = _db_exists(name)
        out.append({"name": name, "type": k, "exists": exists})
    _dbg("_predicted_candidates_for", base=base_db, candidates=out)
    return out

# ------------------------
# Routes
# ------------------------

@router.post("/run", response_model=LPResponse)
def lp_run(body: LPBody):
    _dbg("lp_run.endpoint", body=body.dict())
    return _run_lp_logic(body)

@router.post("/train", response_model=LPResponse)
def lp_train(body: LPBody):
    _dbg("lp_train.endpoint", body=body.dict())
    return _run_lp_logic(body)

@router.get("/predicted/dbs")
def lp_predicted_dbs():
    base = current_database()
    _dbg("lp_predicted_dbs.endpoint", base=base)
    return {"base": base, "candidates": _predicted_candidates_for(base)}

@router.get("/predicted/dbs/{base}")
def lp_predicted_dbs_for(base: str):
    _dbg("lp_predicted_dbs_for.endpoint", base=base)
    return {"base": base, "candidates": _predicted_candidates_for(base)}

@router.get("/predicted/edges")
def lp_predicted_edges():
    """
    Return predicted edge sets keyed by embedding family.
    """
    base = _ensure_cache_for_current_db()
    _dbg("lp_predicted_edges.endpoint", base=base)
    out: dict[str, list[dict[str, str]]] = {}

    for k in _EMBED_KEYS:
        lst = _PREDICTED_EDGES.get(k) or []
        if lst:
            out[k] = [{"s": s, "t": t} for (s, t) in lst]

    for k in _EMBED_KEYS:
        if k in out and out[k]:
            continue
        dbname = f"{base}-predicted-{k}"
        if _db_exists(dbname):
            pairs = _load_predicted_pairs_from_db(dbname)
            if pairs:
                out[k] = [{"s": s, "t": t} for (s, t) in pairs]

    _dbg("lp_predicted_edges.done", keys=list(out.keys()))
    return {"base": base, "edges": out}

@router.get("/test-edges")
def lp_test_edges():
    """
    Return the (canonical) test-set edge list for the current DB, based on the
    default 'testGraph' GDS projection.
    Shape:
      {
        "base": "<dbname>",
        "edges": [ { "s": "...", "t": "..." }, ... ]
      }
    """
    base = _ensure_cache_for_current_db()
    pairs = _get_test_pairs("testGraph")
    return {"base": base, "edges": [{"s": s, "t": t} for (s, t) in pairs]}


@router.get("/predicted/edges/by-db")
def lp_predicted_edges_by_db():
    """
    Explicitly read predicted edges from all available materialized predicted DBs.
    """
    base = current_database()
    _dbg("lp_predicted_edges_by_db.endpoint", base=base)
    out: dict[str, list[dict[str, str]]] = {}
    for cand in _predicted_candidates_for(base):
        if cand.get("exists"):
            pairs = _load_predicted_pairs_from_db(cand["name"])
            if pairs:
                out[cand["type"]] = [{"s": s, "t": t} for (s, t) in pairs]
    _dbg("lp_predicted_edges_by_db.done", keys=list(out.keys()))
    return {"base": base, "edges": out}

@router.get("/metrics")
def lp_metrics():
    """
    Return last-run metrics per embedding family for the current base DB.
    Shape:
      {
        "base": "<dbname>",
        "metrics": {
          "FastRP": { "train": {...}, "validation": {...}|null, "test": {...}|null },
          ...
        }
      }
    """
    base = _ensure_cache_for_current_db()
    _dbg("lp_metrics.endpoint", base=base, keys=list(_METRICS_BY_KEY.keys()))
    return {"base": base, "metrics": _METRICS_BY_KEY}

@router.get("/predicted/timestamps/{dbname}")
def lp_predicted_timestamps(dbname: str):
    """
    For the given predicted DB, return:
      - maxPredicted: max() of the numeric 'predicted' property on relationships
      - maxTimestamp: max() of the 'timestamp' property on relationships

    If there are no relationships, both are 0.
    """
    try:
        rows = run_data(
            """
            MATCH ()-[r]-()
            WITH
              CASE WHEN r.predicted IS NULL THEN 0 ELSE toInteger(r.predicted) END AS lvl,
              CASE WHEN r.timestamp IS NULL THEN 0 ELSE toInteger(r.timestamp) END AS ts
            RETURN
              coalesce(max(lvl), 0) AS maxPredicted,
              coalesce(max(ts), 0)  AS maxTimestamp
            """,
            db=dbname,
        )
    except Exception as e:
        _dbg("lp_predicted_timestamps.error", dbname=dbname, error=str(e))
        raise HTTPException(
            status_code=400,
            detail=f"Failed to inspect predicted DB '{dbname}': {e}",
        )

    row = rows[0] if rows else {"maxPredicted": 0, "maxTimestamp": 0}
    return {
        "db": dbname,
        "maxPredicted": int(row.get("maxPredicted", 0) or 0),
        "maxTimestamp": int(row.get("maxTimestamp", 0) or 0),
    }
