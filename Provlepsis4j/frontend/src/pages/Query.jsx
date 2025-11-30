// src/pages/Query.jsx
import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  queryBoth,
  queryBothGraph,
  getQueryDbs,
  /* optional */ getPredictedDbOptions,
} from "../components/api";
import GraphCanvas from "../components/GraphCanvas.jsx";

/* ---- shared button style to match "Administrator" / "User" ---- */
const viewBtnStyle = (active) => ({
  height: 32,
  border: `1px solid ${active ? "#111827" : "#ccc"}`,
  borderRadius: 4,
  background: "#f8f8f8",
  padding: "0 12px",
  fontSize: 12,
  fontWeight: active ? 600 : 500,
  color: "#111827",
  cursor: "pointer",
  flexShrink: 0,
  display: "inline-flex",
  alignItems: "center",
  justifyContent: "center",
});

/* ---- shared title style (matches SidePanel / DB titles) ---- */
const headingStyle = {
  fontSize: 18,
  fontWeight: 700,
  color: "#111827",
  marginBottom: 4,
};

/* ---------- client-side fallback graphify ---------- */
function jsGraphify(records = []) {
  const nodes = new Map();
  const edges = new Map();

  const asId = (x) => (x == null ? "" : String(x));

  const addNode = (n) => {
    if (!n) return;
    const id =
      asId(
        n.identity ?? n.id ?? n.element_id ?? n.elementId ?? n.elementID
      );
    if (!id) return;
    if (!nodes.has(id)) {
      nodes.set(id, {
        id,
        labels: Array.isArray(n.labels) ? n.labels : [],
        ...(n.properties || {}),
      });
    }
  };

  const addRel = (r) => {
    if (!r) return;
    const rid = asId(
      r.identity ??
        r.id ??
        `${r.start || r.source}->${r.end || r.target}:${r.type || ""}`
    );
    const type = r.type || "";
    const srcId = asId(r.start ?? r.startNode ?? r.source);
    const dstId = asId(r.end ?? r.endNode ?? r.target);
    if (!srcId || !dstId) return;
    addNode({ id: srcId, labels: [], properties: {} });
    addNode({ id: dstId, labels: [], properties: {} });
    edges.set(rid, {
      id: rid,
      type,
      source: srcId,
      target: dstId,
      ...(r.properties || {}),
    });
  };

  const eat = (v) => {
    if (!v) return;

    // Path {segments:[...]}
    if (Array.isArray(v?.segments) && v.segments.length) {
      v.segments.forEach((s) => {
        if (s?.start) addNode(s.start);
        if (s?.end) addNode(s.end);
        if (s?.relationship) addRel(s.relationship);
      });
      if (v?.end) addNode(v.end);
      return;
    }
    // Path {nodes, relationships}
    if (Array.isArray(v?.nodes) && Array.isArray(v?.relationships)) {
      v.nodes.forEach(addNode);
      v.relationships.forEach(addRel);
      return;
    }
    // Node-like
    if (
      (v.identity ?? v.id ?? v.element_id ?? v.elementId ?? v.elementID) &&
      v.labels &&
      v.properties
    ) {
      addNode(v);
      return;
    }
    // Relationship-like
    if (
      (v.identity ?? v.id) &&
      v.type &&
      (v.start ?? v.startNode ?? v.source) &&
      (v.end ?? v.endNode ?? v.target)
    ) {
      addRel(v);
      return;
    }
    // Arrays
    if (Array.isArray(v)) {
      v.forEach(eat);
      return;
    }
    // Objects
    if (typeof v === "object") {
      Object.values(v).forEach(eat);
    }
  };

  for (const row of records) {
    Object.values(row || {}).forEach(eat);
    // Common RETURN a,b,r pattern
    const a = row?.a,
      b = row?.b,
      r = row?.r;
    if (a && b && r) {
      addNode(a);
      addNode(b);
      addRel(r);
    }
  }

  return {
    nodes: Array.from(nodes.values()),
    edges: Array.from(edges.values()),
  };
}

/* ---------- simple table helpers ---------- */
function prettyVal(v) {
  if (v == null) return v;
  if (typeof v === "object") {
    const o = v;
    if ((o.identity || o.id || o.element_id) && o.labels && o.properties) {
      return {
        id: o.identity ?? o.id ?? o.element_id,
        labels: o.labels,
        ...o.properties,
      };
    }
    if (
      (o.identity || o.id) &&
      o.type &&
      (o.start || o.startNode) &&
      (o.end || o.endNode)
    ) {
      return {
        id: o.identity ?? o.id,
        type: o.type,
        start: o.start ?? o.startNode,
        end: o.end ?? o.endNode,
        ...o.properties,
      };
    }
    if (Array.isArray(o.segments)) {
      return {
        nodes: o.segments
          .map((s) => s.start?.properties)
          .concat(o.end?.properties ?? []),
        rels: o.segments.map((s) => ({
          type: s.relationship?.type,
          ...s.relationship?.properties,
        })),
      };
    }
  }
  return v;
}

function toTable(records = []) {
  const columns = Array.from(
    records.reduce((acc, r) => {
      Object.keys(r).forEach((k) => acc.add(k));
      return acc;
    }, new Set())
  );
  const rows = records.map((r) => columns.map((c) => prettyVal(r[c])));
  return { columns, rows };
}

function ResultTable({ records }) {
  const data = useMemo(() => toTable(records || []), [records]);
  if (!records || records.length === 0)
    return (
      <div style={{ opacity: 0.7, fontSize: 13 }}>(No rows)</div>
    );
  return (
    <div
      style={{
        overflow: "auto",
        border: "1px solid #e5e7eb",
        borderRadius: 12,
      }}
    >
      <table
        style={{
          width: "100%",
          fontSize: 13,
          borderCollapse: "collapse",
        }}
      >
        <thead style={{ background: "#f9fafb" }}>
          <tr>
            {data.columns.map((c) => (
              <th
                key={c}
                style={{
                  textAlign: "left",
                  padding: "6px 8px",
                  borderBottom: "1px solid #e5e7eb",
                }}
              >
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.rows.map((row, i) => (
            <tr
              key={i}
              style={{ background: i % 2 ? "#fafafa" : "#fff" }}
            >
              {row.map((cell, j) => (
                <td
                  key={j}
                  style={{
                    verticalAlign: "top",
                    padding: "6px 8px",
                    borderBottom: "1px solid #f1f5f9",
                  }}
                >
                  <pre
                    style={{
                      margin: 0,
                      whiteSpace: "pre-wrap",
                      wordBreak: "break-word",
                    }}
                  >
                    {JSON.stringify(cell, null, 2)}
                  </pre>
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ---------- per-panel view with optional right controls ---------- */
function ResultView({ title, loading, graph, records, rightControls }) {
  const [view, setView] = useState("graph"); // "graph" | "table"
  const hasGraph = graph?.nodes?.length || graph?.edges?.length;

  return (
    <section
      style={{
        display: "grid",
        gridTemplateRows: "auto 1fr",
        gap: 8,
        border: "1px solid #e5e7eb",
        borderRadius: 12,
        padding: 12,
        minHeight: 0,
        minWidth: 0,
      }}
    >
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          gap: 8,
        }}
      >
        <strong style={headingStyle}>{title}</strong>

        {/* right-side controls (dropdown lives here on the future panel) */}
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          {rightControls}
          <div
            style={{ display: "flex", gap: 8, alignItems: "center" }}
          >
            {loading && (
              <span style={{ fontSize: 12, opacity: 0.7 }}>
                Running…
              </span>
            )}
            <button
              onClick={() => setView("graph")}
              style={viewBtnStyle(view === "graph")}
            >
              Graph
            </button>
            <button
              onClick={() => setView("table")}
              style={viewBtnStyle(view === "table")}
            >
              Table
            </button>
          </div>
        </div>
      </div>

      <div style={{ minHeight: 0, minWidth: 0 }}>
        {view === "graph"
          ? hasGraph
            ? (
              <GraphCanvas graph={graph} height={420} />
              )
            : null
          : <ResultTable records={records} />}
      </div>
    </section>
  );
}

/* ===================== Main page ===================== */
export default function Query() {
  const [cypher, setCypher] = useState(
    "MATCH p=(a)-[r]->(b) RETURN p LIMIT 25"
  );
  const [threshold, setThreshold] = useState(0);
  const [present, setPresent] = useState({
    records: [],
    graph: { nodes: [], edges: [] },
  });
  const [future, setFuture] = useState({
    records: [],
    graph: { nodes: [], edges: [] },
  });
  const [dbs, setDbs] = useState({ present: "", predicted: "" });

  // new state for predicted-DB selector
  const [predictedChoices, setPredictedChoices] = useState([]); // [{name,type,exists}]
  const [futureDb, setFutureDb] = useState(""); // selected predicted DB
  // Broadcast currently selected predicted DB so the SidePanel can react
  useEffect(() => {
    if (!futureDb) return;
    window.dispatchEvent(
      new CustomEvent("neo4j:predicted-db-changed", {
        detail: { futureDb },
      })
    );
  }, [futureDb]);

  const [loading, setLoading] = useState({ p: false, f: false });
  const [err, setErr] = useState("");

  // load base/predicted defaults on first mount
  useEffect(() => {
    getQueryDbs()
      .then((d) => d && setDbs(d))
      .catch(() => {});
  }, []);

  // refetch the base/predicted DB labels whenever someone broadcasts a db change
  useEffect(() => {
    const onDbChanged = () => {
      getQueryDbs()
        .then((d) => d && setDbs(d))
        .catch(() => {});
    };
    window.addEventListener("neo4j:db-changed", onDbChanged);
    return () =>
      window.removeEventListener("neo4j:db-changed", onDbChanged);
  }, []);

  // load predicted DB options when present DB is known
  useEffect(() => {
    const deriveCandidates = (base) => {
      if (!base) return [];
      return ["Node2Vec", "FastRP", "GraphSAGE", "HashGNN"].map(
        (t) => ({
          name: `${base}-predicted-${t}`,
          type: t,
          exists: true,
        })
      );
    };

    (async () => {
      try {
        if (typeof getPredictedDbOptions === "function") {
          const { base, candidates } = await getPredictedDbOptions();
          const list =
            candidates && candidates.length
              ? candidates
              : deriveCandidates(base || dbs.present);
          setPredictedChoices(list);
          const firstExisting = list.find((c) => c.exists);
          setFutureDb(
            firstExisting?.name || list[0]?.name || ""
          );
        } else {
          const list = deriveCandidates(dbs.present);
          setPredictedChoices(list);
          setFutureDb(list[0]?.name || "");
        }
      } catch {
        const list = deriveCandidates(dbs.present);
        setPredictedChoices(list);
        setFutureDb(list[0]?.name || "");
      }
    })();
  }, [dbs.present]);

  const runBoth = useCallback(
    async () => {
      setErr("");
      setLoading({ p: true, f: true });
      try {
        const g = await queryBothGraph({
          cypher,
          params: { thr: Number(threshold) },
          futureDb,
        });
        const presentGraph =
          g?.present?.nodes?.length || g?.present?.edges?.length
            ? g.present
            : null;
        const predictedGraph =
          g?.predicted?.nodes?.length ||
          g?.predicted?.edges?.length
            ? g.predicted
            : null;

        const t = await queryBoth({
          cypher,
          params: { thr: Number(threshold) },
          futureDb,
        });

        const pGraph = presentGraph ?? jsGraphify(t?.present || []);
        const fGraph =
          predictedGraph ??
          jsGraphify(t?.predicted || t?.future || []);

        setPresent({
          records: t?.present || [],
          graph: pGraph,
        });
        setFuture({
          records: t?.predicted || t?.future || [],
          graph: fGraph,
        });

        const dbLabels =
          g?.databases ||
          t?.databases || {
            present: dbs.present,
            predicted: futureDb || dbs.predicted,
          };
        setDbs(dbLabels);
      } catch (e) {
        setErr(e?.response?.data?.detail || e.message);
      } finally {
        setLoading({ p: false, f: false });
      }
    },
    [cypher, threshold, futureDb, dbs.present]
  );

  return (
    <div
      style={{
        padding: 16,
        display: "grid",
        gap: 12,
        height: "100%",
        gridTemplateRows: "auto 1fr",
      }}
    >
      <div style={{ display: "grid", gap: 8 }}>
        {/* TITLE NOW MATCHES OTHER SECTION HEADINGS */}
        <label style={headingStyle}>Cypher Query</label>

        <textarea
          value={cypher}
          onChange={(e) => setCypher(e.target.value)}
          placeholder="Write any Cypher. Left runs on present DB, right on selected predicted DB."
          style={{
            height: 140,
            padding: 12,
            border: "1px solid #d1d5db",
            borderRadius: 12,
            fontFamily:
              "ui-monospace, SFMono-Regular, Menlo, monospace",
          }}
        />

        <div
          style={{ display: "flex", justifyContent: "flex-end" }}
        >
          <button
            onClick={runBoth}
            style={{ ...viewBtnStyle(true), height: 40 }}
          >
            Run on both
          </button>
        </div>
        {err && <div style={{ color: "#b91c1c" }}>{err}</div>}
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: 16,
          minHeight: 0,
        }}
      >
        <div style={{ minWidth: 0 }}>
          <ResultView
            title={`Database: ${dbs.present || "present DB"}`}
            loading={loading.p}
            graph={present.graph}
            records={present.records}
          />
        </div>

        <div style={{ minWidth: 0 }}>
          <ResultView
            title={`Database: ${futureDb ||
              dbs.predicted ||
              "predicted DB"}`}
            loading={loading.f}
            graph={future.graph}
            records={future.records}
            rightControls={
              <select
                value={futureDb}
                onChange={(e) => setFutureDb(e.target.value)}
                style={{
                  padding: "4px 8px",
                  borderRadius: 6,
                  border: "1px solid #e5e7eb",
                }}
                title="Pick the predicted DB variant"
              >
                {predictedChoices.map(({ name, exists }) => (
                  <option
                    key={name}
                    value={name}
                    disabled={exists === false}
                  >
                    {name}
                    {exists === false ? " (missing)" : ""}
                  </option>
                ))}
              </select>
            }
          />
        </div>
      </div>
    </div>
  );
}
