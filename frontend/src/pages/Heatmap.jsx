// src/pages/Heatmap.jsx
import React, { useEffect, useMemo, useState, useCallback } from "react";
import {
  getPredictedDbOptions,
  getPredictedEdgeSets,
  getLpMetrics,
  getLpGroundTruthEdges
} from "../components/api";

// ---------- typography tiers ----------
const FONT_TITLE = { fontSize: 18, fontWeight: 700 };        
const FONT_SUBTITLE = { fontSize: 15, fontWeight: 600 };     
const FONT_BODY = { fontSize: 14, fontWeight: 400, color: "#111827" }; 

// tiny color mapper 0..100 → classic white→red heatmap
function cellColor(pct) {
  const t = Math.max(0, Math.min(1, pct / 100)); 
  const r = 255;
  const g = Math.round(255 * (1 - t));
  const b = Math.round(255 * (1 - t));
  return `rgb(${r}, ${g}, ${b})`;
}

function pct(v) {
  return (Math.round(v * 10000) / 100).toFixed(2);
}

function fmtMetric(x) {
  if (x == null || Number.isNaN(x)) return "—";
  return Number(x).toFixed(3);
}

const ALL_KEYS = ["Node2Vec", "FastRP", "GraphSAGE", "HashGNN"];

export default function Heatmap() {
  // availability of predicted DB variants for current base db
  const [choices, setChoices] = useState([]); // [{name,type,exists}]
  // selected embeddings
  const [selected, setSelected] = useState(new Set());
  // edges per type (canonical "s|t" strings)
  const [edgeSets, setEdgeSets] = useState({}); // {type: Set<string>}
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");

  // metrics per embedding
  const [metrics, setMetrics] = useState({}); 

  // view toggles
  const [showMetrics, setShowMetrics] = useState(true);
  const [showHeatmap, setShowHeatmap] = useState(true);

  // ground truth edges
  const [groundTruth, setGroundTruth] = useState(new Set());
  const [showGroundTruth, setShowGroundTruth] = useState(true);

  // ----- load which variants exist & default-select all existing -----
  useEffect(() => {
    (async () => {
      try {
        const { candidates } = await getPredictedDbOptions();
        setChoices(candidates || []);
        const enabled = new Set(
          (candidates || []).filter((c) => c.exists).map((c) => c.type)
        );
        setSelected(enabled); 
      } catch {
        setChoices([]);
      }
    })();
  }, []);

  // ----- fetch predicted edges -----
  const loadEdges = useCallback(async () => {
    try {
      const { edges } = await getPredictedEdgeSets();
      const canon = {};
      for (const key of Object.keys(edges || {})) {
        const pairs = edges[key] || [];
        const s = new Set(
          pairs.map(({ s, t }) => {
            const a = String(s || "");
            const b = String(t || "");
            return a < b ? `${a}|${b}` : `${b}|${a}`;
          })
        );
        canon[key] = s;
      }
      setEdgeSets(canon);
    } catch (e) {
      setErr(
        e?.response?.data?.detail ||
          e.message ||
          "Failed to load predicted edges."
      );
    }
  }, []);


  // ----- fetch metrics -----
  const loadMetrics = useCallback(async () => {
    try {
      const { metrics } = await getLpMetrics();
      setMetrics(metrics || {});
    } catch {
      setMetrics({});
    }
  }, []);

  // ----- fetch Ground Truth (test-set) edges -----
  const loadGroundTruth = useCallback(async () => {
    try {
      const { edges } = await getLpGroundTruthEdges();
      const s = new Set(
        (edges || []).map(({ s, t }) => {
          const a = String(s || "");
          const b = String(t || "");
          return a < b ? `${a}|${b}` : `${b}|${a}`;
        })
      );
      setGroundTruth(s);
    } catch (e) {
      console.error("Failed to load Ground Truth edges", e);
    }
  }, []);


  // refresh both components (edges + metrics)
  const refreshAll = useCallback(async () => {
    setLoading(true);
    setErr("");
    try {
      await Promise.all([loadEdges(), loadMetrics(), loadGroundTruth()]);
    } finally {
      setLoading(false);
    }
  }, [loadEdges, loadMetrics, loadGroundTruth]);

  // first load
  useEffect(() => {
    refreshAll();
  }, [refreshAll]);

  // ---- refresh when the active DB changes (broadcast from elsewhere) ----
  useEffect(() => {
    const onDbChanged = () => {
      getPredictedDbOptions()
        .then(({ candidates }) => {
          setChoices(candidates || []);
          const enabled = new Set(
            (candidates || []).filter((c) => c.exists).map((c) => c.type)
          );
          setSelected(enabled);
          refreshAll();
        })
        .catch(() => {});
    };
    window.addEventListener("neo4j:db-changed", onDbChanged);
    return () => window.removeEventListener("neo4j:db-changed", onDbChanged);
  }, [refreshAll]);


 const matrix = useMemo(() => {
  const baseKeys = ALL_KEYS.filter((k) => selected.has(k));
  const hasEmbeddings = baseKeys.length > 0;

  // Only show Ground Truth if there is at least one embedding selected
  let keys = baseKeys;
  if (showGroundTruth && hasEmbeddings) {
    keys = [...baseKeys, "Ground Truth"];
  }

  // If no embeddings, do not render any matrix (handles "only GT" case)
  if (!hasEmbeddings) return { keys: [], data: [] };

  const getSetForKey = (k) => {
    if (k === "Ground Truth") {
      return groundTruth || new Set();
    }
    return edgeSets[k] || new Set();
  };

  const data = keys.map((rowK) => {
    const rowSet = getSetForKey(rowK);
    return keys.map((colK) => {
      const colSet = getSetForKey(colK);

      // Ground Truth vs Ground Truth → 100% if we actually have GT edges
      if (rowK === "Ground Truth" && colK === "Ground Truth") {
        return groundTruth && groundTruth.size > 0 ? 100 : 0;
      }

      if (rowSet.size === 0 || colSet.size === 0) return 0;

      const k = Math.min(rowSet.size, colSet.size);
      let common = 0;
      const iter = rowSet.size <= colSet.size ? rowSet : colSet;
      const other = rowSet.size <= colSet.size ? colSet : rowSet;
      for (const p of iter) if (other.has(p)) common++;
      return (common / (k || 1)) * 100;
    });
  });

  return { keys, data };
}, [selected, edgeSets, groundTruth, showGroundTruth]);



  const toggleEmbedding = (k, enabled) => {
    const next = new Set(selected);
    if (enabled) next.add(k);
    else next.delete(k);
    setSelected(next);
  };

  const visibleKeys = ALL_KEYS.filter((k) => selected.has(k));

  return (
    <div
      style={{
        height: "100%",
        display: "grid",
        gridTemplateColumns: "300px 1fr",
        gap: 12,
      }}
    >
      {/* Left side panel */}
      <aside style={{ borderRight: "1px solid #e5e7eb", paddingRight: 12 }}>
        {/* Components card (toggles for metrics / heatmap) */}
        <div
          style={{
            borderRadius: 8,
            border: "1px solid #e5e7eb",
            padding: 12,
            background: "#f9fafb",
          }}
        >
        <div
          style={{
            ...FONT_TITLE,
            marginBottom: 8,
          }}
        >
            Evaluation
          </div>

          <label
            style={{
              display: "flex",
              alignItems: "center",
              gap: 8,
              marginBottom: 4,
              ...FONT_BODY,
            }}
          >
            <input
              type="checkbox"
              checked={showMetrics}
              onChange={(e) => setShowMetrics(e.target.checked)}
            />
            <span>Performance Metrics</span>
          </label>
          <label
            style={{
              display: "flex",
              alignItems: "center",
              fontSize: 18,
              gap: 8,
              ...FONT_BODY,
            }}
          >
            <input
              type="checkbox"
              checked={showHeatmap}
              onChange={(e) => setShowHeatmap(e.target.checked)}
            />
            <span>Prediction Comparison</span>
          </label>
        </div>


        {/* Encased embeddings card */}
        <div
          style={{
            borderRadius: 8,
            border: "1px solid #e5e7eb",
            padding: 12,
            marginTop: 8,
            marginBottom: 12,
            background: "#f9fafb",
          }}
        >
          <h3
            style={{
              ...FONT_TITLE,
              margin: 0,
              marginBottom: 12,
            }}
          >
            Prediction Comparison
          </h3>


          <div style={{ display: "grid", gap: 8 }}>
            {ALL_KEYS.map((k) => {
              const choice = choices.find((c) => c.type === k);
              const exists = choice ? !!choice.exists : false;
              const checked = selected.has(k);
              return (
                <label
                  key={k}
                  style={{
                    opacity: exists ? 1 : 0.45,
                    display: "flex",
                    gap: 8,
                    alignItems: "center",
                    ...FONT_BODY,
                  }}
                >
                  <input
                    type="checkbox"
                    disabled={!exists}
                    checked={exists ? checked : false}
                    onChange={(e) => toggleEmbedding(k, e.target.checked)}
                  />
                  <span>{k}</span>
                </label>
              );
            })}

            {/* Ground Truth overlap toggle inside the same square */}
            <label
              style={{
                display: "flex",
                gap: 8,
                alignItems: "center",
                ...FONT_BODY,
              }}
            >
              <input
                type="checkbox"
                checked={showGroundTruth}
                onChange={(e) => setShowGroundTruth(e.target.checked)}
              />
              <span>Ground Truth</span>
              {/* <span style={{ fontSize: 12, color: "#6b7280" }}>
                {groundTruth.size > 0 ? `${groundTruth.size} edges` : "— no test edges"}
              </span> */}
            </label>
          </div>

        </div>

 

        <div style={{ height: 12 }} />

        {/* Refresh button  */}
        <button
          onClick={refreshAll}
          disabled={loading}
          style={{
            width: "100%",
            height: 32,
            borderRadius: 4,
            border: "1px solid #d1d5db",
            background: "#f9fafb",
            cursor: loading ? "default" : "pointer",
            ...FONT_BODY,
          }}
          title="Recompute metrics and overlap from current predicted graphs"
        >
          {loading ? "Refreshing…" : "Refresh Components"}
        </button>

        {err && (
          <div style={{ color: "#b91c1c", marginTop: 8, ...FONT_BODY }}>
            {err}
          </div>
        )}
      </aside>

      {/* Main: metrics (top) + heatmap (bottom) */}
      <section
        style={{
          padding: 8,
          overflow: "auto",
          display: "flex",
          flexDirection: "column",
          gap: 16,
        }}
      >
        {/* Performance metrics (upper area) */}
        {showMetrics && visibleKeys.length > 0 && (
          <div
            style={{
              borderRadius: 12,
              border: "1px solid #e5e7eb",
              background: "#f9fafb",
              padding: 12,
              display: "flex",
              flexDirection: "column",
              fontSize: 18,
              gap: 12,
            }}
          >
            <div style={{ ...FONT_TITLE}}>Performance Metrics</div>

            <div
              style={{
                display: "flex",
                gap: 12,
                flexWrap: "wrap",
              }}
            >
              {visibleKeys.map((key) => {

                const m = metrics[key]?.test || metrics[key]?.validation || metrics[key]?.train || metrics[key];
                if (!m) return null;

                const rows = [
                  { label: "Accuracy", value: m.accuracy },
                  { label: "Precision", value: m.precision },
                  { label: "Recall", value: m.recall },
                  { label: "F1 Score", value: m.f1 },
                  { label: "AUC", value: m.auc },
                ];

                return (
                  <div
                    key={key}
                    style={{
                      flex: "1 1 0",
                      minWidth: 200,
                      maxWidth: 320,
                      borderRadius: 8,
                      border: "1px solid #e5e7eb",
                      background: "#ffffff",
                      padding: 10,
                      display: "flex",
                      flexDirection: "column",
                      gap: 4,
                      boxShadow: "0 1px 2px rgba(0,0,0,0.03)",
                    }}
                  >
                    <div
                      style={{
                        ...FONT_SUBTITLE, 
                        marginBottom: 4,
                      }}
                    >
                      {key}
                    </div>
                    {rows.map((r) => (
                      <div
                        key={r.label}
                        style={{
                          ...FONT_BODY,
                          display: "flex",
                          justifyContent: "space-between",
                        }}
                      >
                        <span>{r.label}</span>
                        <span style={{ fontWeight: 600 }}>
                          {fmtMetric(r.value)}
                        </span>
                      </div>
                    ))}
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Heatmap (lower area) */}
        {showHeatmap && (
          <div
            style={{
              borderRadius: 12,
              border: "1px solid #e5e7eb",
              background: "#ffffff",
              padding: 12,
              flex: 1,
              overflow: "auto",
            }}
          >
            <div
              style={{
                display: "flex",
                alignItems: "baseline",
                gap: 12,
                marginBottom: 8,
              }}
            >
              <div style={{ ...FONT_TITLE}}>Prediction Comparison</div>
            </div>

            {matrix.keys.length === 0 ? (
              <div style={{ marginTop: 12, ...FONT_BODY }}>
                Select at least one embedding with predictions.
              </div>
            ) : (
              <div style={{ overflow: "auto" }}>
                <table style={{ borderCollapse: "collapse" }}>
                  <thead>
                    <tr>
                      <th />
                      {matrix.keys.map((k) => (
                        <th
                          key={k}
                          style={{
                            padding: "6px 8px",
                            textAlign: "center",
                            ...FONT_SUBTITLE, 
                          }}
                        >
                          {k}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {matrix.data.map((row, i) => (
                      <tr key={matrix.keys[i]}>
                        <th
                          style={{
                            padding: "6px 8px",
                            textAlign: "right",
                            whiteSpace: "nowrap",
                            ...FONT_SUBTITLE, 
                          }}
                        >
                          {matrix.keys[i]}
                        </th>
                        {row.map((v, j) => (
                          <td
                            key={j}
                            style={{
                              padding: 0,
                              border: "1px solid #e5e7eb",
                              width: 72,
                              height: 40,
                              background: cellColor(v),
                              textAlign: "center",
                              fontWeight: 600,
                              ...FONT_BODY, 
                            }}
                          >
                            {pct(v / 100)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
      </section>
    </div>
  );
}
