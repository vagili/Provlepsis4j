import React, { useState, useEffect } from "react";
import {
  api,
  runLP,
  executeSplit,
  uploadGraph,
  setNeo4jConfig,
  isConfigured,
  markConfigured,
  clearConfigured,
  iteratePrediction,
  getPredictedTimestampCount,
} from "./api";
import LoadDatasetModal from "./LoadDatasetModal.jsx";
import { useGraph } from "../graphContext.jsx";

const EMBEDDINGS = ["FastRP", "Node2Vec", "GraphSAGE", "HashGNN"];

// shared field style for text/select/number
const baseField = {
  width: "100%",
  height: 32,
  padding: "4px 8px",
  border: "1px solid #ccc",
  borderRadius: 4,
  boxSizing: "border-box",
};

const s = {
  panel: {
    width: 320,
    borderRight: "1px solid #ddd",
    padding: 12,
    overflow: "auto",
    display: "grid",
    gap: 16,
    boxSizing: "border-box",
  },
  section: { display: "grid", gap: 10 },
  h3: { margin: 0, fontSize: 18, fontWeight: 700, color: "#111827" },
  label: { fontSize: 12, opacity: 0.85, display: "block", marginBottom: 4 },

  // unified field styles
  input: baseField,
  number: baseField,

  btn: {
    height: 32,
    border: "1px solid #ccc",
    borderRadius: 4,
    background: "#f8f8f8",
    padding: "0 12px",
    cursor: "pointer",
  },
  btnDisabled: {
    height: 32,
    border: "1px solid #ddd",
    borderRadius: 4,
    background: "#eee",
    opacity: 0.7,
    cursor: "not-allowed",
    padding: "0 12px",
  },
  log: {
    whiteSpace: "pre-wrap",
    background: "#f7f7f7",
    padding: 8,
    minHeight: 70,
    borderRadius: 4,
  },
};

export default function SidePanel({ page, setPage }) {
  const { graphName } = useGraph();

  // ======= Load Dataset =======
  const [openLoad, setOpenLoad] = useState(false);

  // ===== Split Graph =====
  const [testHoldout, setTestHoldout] = useState(0.1);
  const [valHoldout, setValHoldout] = useState(0.0);
  const [splitLog, setSplitLog] = useState("");
  const [splitBusy, setSplitBusy] = useState(false);

  // ===== Embeddings =====
  const [selectedEmb, setSelectedEmb] = useState("FastRP");
  const [embLog, setEmbLog] = useState("");
  const [embBusy, setEmbBusy] = useState(false);

  // Keep LP embedding property in sync with the dropdown
  const [embeddingProperty, setEmbeddingProperty] = useState("FastRP");
  useEffect(() => {
    setEmbeddingProperty(selectedEmb);
  }, [selectedEmb]);

  // ===== Train + Predict (LP) =====
  const [predictK, setPredictK] = useState(100);
  const negativeRatio = 1;
  const [probThreshold, setProbThreshold] = useState(0.5);
  const [outputGraphName, setOutputGraphName] = useState("predictedGraph");
  const [lpLog, setLpLog] = useState("");
  const [lpBusy, setLpBusy] = useState(false);

  async function runEmbedding() {
    setEmbBusy(true);
    setEmbLog(`Running ${selectedEmb}...`);
    const wp = selectedEmb; // property name mirrors the chosen embedding
    const g = graphName || undefined;
    try {
      if (selectedEmb === "FastRP") {
        const { data } = await api.post(
          "/emb/fastrp/write",
          { graphName: g, writeProperty: wp, embeddingDimension: 128, propertyRatio: 0.5 },
          { timeout: 0 }
        );
        setEmbLog(JSON.stringify(data, null, 2));
      } else if (selectedEmb === "Node2Vec") {
        const { data } = await api.post(
          "/emb/node2vec/write",
          { graphName: g, writeProperty: wp, embeddingDimension: 128 },
          { timeout: 0 }
        );
        setEmbLog(JSON.stringify(data, null, 2));
      } else if (selectedEmb === "GraphSAGE") {
        const { data } = await api.post(
          "/emb/graphsage/trainWrite",
          { graphName: g, writeProperty: wp, embeddingDimension: 128 },
          { timeout: 0 }
        );
        setEmbLog(JSON.stringify(data, null, 2));
      } else if (selectedEmb === "HashGNN") {
        const { data } = await api.post(
          "/emb/hashgnn/write",
          { graphName: g, writeProperty: wp, embeddingDimension: 128 },
          { timeout: 0 }
        );
        setEmbLog(JSON.stringify(data, null, 2));
      }
    } catch (e) {
      setEmbLog(
        typeof e?.response?.data === "object"
          ? JSON.stringify(e.response.data, null, 2)
          : e?.response?.data?.detail || e.message || String(e)
      );
    } finally {
      setEmbBusy(false);
    }
  }

  async function runSplit() {
    setSplitBusy(true);
    setSplitLog("Splitting graph...");
    const body = {
      trainGraphName: "trainGraph",
      testGraphName: "testGraph",
      valGraphName: "valGraph",
      testHoldout: Number(testHoldout),
      valHoldout: Number(valHoldout),
    };
    try {
      const data = await executeSplit(body);
      setSplitLog(JSON.stringify(data, null, 2));
      window.dispatchEvent(new Event("neo4j:db-changed"));
    } catch (e) {
      setSplitLog(String(e?.response?.data || e.message || e));
    } finally {
      setSplitBusy(false);
    }
  }

  async function runTrainPredict() {
    setLpBusy(true);
    setLpLog("Training + Predicting...");
    try {
      const res = await runLP(
        {
          embeddingProperty,
          predictK: Number(predictK),
          negativeRatio: Number(negativeRatio),
          probThreshold: Number(probThreshold),
          outputGraphName,
        },
        { timeout: 0 }
      );
      setLpLog(JSON.stringify(res, null, 2));
    } catch (e) {
      setLpLog(String(e));
    } finally {
      setLpBusy(false);
    }
  }

  async function runEmbeddingAndPredict() {
    setLpLog("Training + Predicting...");
    try {
      await runEmbedding();       // 1) compute embeddings
      await runTrainPredict();    // 2) run LP using that embedding property
      window.dispatchEvent(new Event("neo4j:db-changed"));
    } catch (e) {
      setLpLog(String(e?.response?.data || e.message || e));
    }
  }

  const Btn = ({ disabled, onClick, children }) => (
    <button onClick={onClick} disabled={disabled} style={disabled ? s.btnDisabled : s.btn}>
      {children}
    </button>
  );

  // ===== NEXT ITERATION box =====
  const [itFamily, setItFamily] = useState("FastRP");
  const [itK, setItK] = useState(100);
  const [itThr, setItThr] = useState(0.5);
  const itNeg = 1;
  const [itMsg, setItMsg] = useState("");
  const [itBusy, setItBusy] = useState(false);

    // ----- User-view: "Number of Predicted Timestamps" counter -----
  const [predictedTsDb, setPredictedTsDb] = useState("");
  const [predictedTsCount, setPredictedTsCount] = useState(null);

  async function fetchPredictedTimestampCount(dbName) {
    if (!dbName) {
      setPredictedTsDb("");
      setPredictedTsCount(null);
      return;
    }
    setPredictedTsDb(dbName);
    try {
      const data = await getPredictedTimestampCount(dbName);
      // Prefer the predicted label count; fall back to 0 if missing
      const n =
        data && typeof data.maxPredicted === "number"
          ? data.maxPredicted
          : 0;
      setPredictedTsCount(n);
    } catch (e) {
      console.error("Failed to load predicted timestamp count", e);
      setPredictedTsCount(null);
    }
  }

  // Listen for changes from Query.jsx's predicted-DB dropdown
  useEffect(() => {
    const handler = (ev) => {
      const name =
        ev?.detail?.futureDb ||
        ev?.detail?.predictedDb ||
        ev?.detail?.db ||
        "";
      if (name) {
        fetchPredictedTimestampCount(name);
      }
    };

    window.addEventListener("neo4j:predicted-db-changed", handler);
    return () =>
      window.removeEventListener("neo4j:predicted-db-changed", handler);
  }, []); // run once on mount

  async function runIteration() {
    setItBusy(true);
    setItMsg("");
    try {
      const out = await iteratePrediction({
        embeddingFamily: itFamily,
        embeddingProperty: itFamily,
        predictK: Number(itK),
        candidateMultiplier: 20,
        probThreshold: Number(itThr),
        negativeRatio: Number(itNeg),
      });
      setItMsg(
        `✓ ${out.added} edges added to ${out.db} (predicted=${out.newPredictedLevel}, ts=${out.newTimestamp})`
      );
      window.dispatchEvent(new Event("neo4j:db-changed"));
      if (predictedTsDb) {
        fetchPredictedTimestampCount(predictedTsDb);
      }
    } catch (e) {
      setItMsg(`✗ ${e?.response?.data?.detail || e.message || e}`);
    } finally {
      setItBusy(false);
    }
  }

  // ---- view toggle button style ----
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
    flex: 1,
  });

  return (
    <aside style={s.panel}>
      {/* Views – always visible */}
      <section>
        <h3 style={s.h3}>Swap Views</h3>
        <div style={{ display: "flex", gap: 8 }}>
          <button
            style={viewBtnStyle(page === "heatmap")}
            onClick={() => setPage?.("heatmap")}
          >
            Administrator
          </button>
          <button
            style={viewBtnStyle(page === "query")}
            onClick={() => setPage?.("query")}
          >
            User
          </button>
        </div>
      </section>

      {/* USER VIEW: Predicted Timestamps Count */}
      {page === "query" && (
        <section>
          <h3 style={s.h3}>Predicted Timestamps</h3>
          <div style={s.section}>
            <div>
              <label style={s.label}>Number of Predicted Timestamps</label>
              <input
                type="number"
                readOnly
                value={predictedTsCount ?? 0}
                style={s.number}
              />
            </div>
          </div>
        </section>
      )}

      {/* USER VIEW: Extend predicted graph directly under Swap Views */}
      {page === "query" && (
        <section>
          <h3 style={s.h3}>Extend Predicted Graph</h3>
          <div style={s.section}>
            <div>
              <label style={s.label}>Embedding</label>
              <select
                value={itFamily}
                onChange={(e) => setItFamily(e.target.value)}
                style={s.input}
              >
                {EMBEDDINGS.map((opt) => (
                  <option key={opt} value={opt}>
                    {opt}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label style={s.label}>Number of Predicted Edges</label>
              <input
                type="number"
                min={1}
                value={itK}
                onChange={(e) => setItK(e.target.value)}
                style={s.number}
              />
            </div>

            <div>
              <label style={s.label}>Edge Existence Probability Threshold</label>
              <input
                type="number"
                step="0.01"
                value={itThr}
                onChange={(e) => setItThr(e.target.value)}
                style={s.number}
              />
            </div>

            {/* now using the same 32px button style */}
            <Btn onClick={runIteration} disabled={itBusy}>
              {itBusy ? "Running…" : "Extend Predicted Graph"}
            </Btn>
            {/* {!!itMsg && <pre style={s.log}>{itMsg}</pre>} */}
          </div>
        </section>
      )}

      {/* ADMIN VIEW: Load dataset + Split + Train & Predict */}
      {page === "heatmap" && (
        <>
          {/* 0. Load dataset – admin only */}
          <section>
            <h3 style={s.h3}>Load Dataset</h3>
            <div style={s.section}>
              <Btn onClick={() => setOpenLoad(true)} disabled={false}>
                Load Dataset
              </Btn>
              <LoadDatasetModal open={openLoad} onClose={() => setOpenLoad(false)} />
            </div>
          </section>

          {/* Split Graph */}
          <section>
            <h3 style={s.h3}>Split Graph</h3>
            <div style={s.section}>
              <div>
                <label style={s.label}>Test Hold-Out </label>
                <input
                  type="number"
                  min={0}
                  max={1}
                  step="0.01"
                  value={testHoldout}
                  onChange={(e) => setTestHoldout(e.target.value)}
                  style={s.number}
                />
              </div>
              <div>
                <label style={s.label}>Validation Hold-Out </label>
                <input
                  type="number"
                  min={0}
                  max={1}
                  step="0.01"
                  value={valHoldout}
                  onChange={(e) => setValHoldout(e.target.value)}
                  style={s.number}
                />
              </div>
              <Btn onClick={runSplit} disabled={splitBusy}>
                {splitBusy ? "Splitting…" : "Split"}
              </Btn>
              {/* <pre style={s.log}>{splitLog}</pre> */}
            </div>
          </section>

          {/* Embeddings + Train & Predict */}
          <section>
            <h3 style={s.h3}>Train &amp; Predict</h3>
            <div style={s.section}>
              <label style={s.label}>Embedding</label>
              <select
                value={selectedEmb}
                onChange={(e) => setSelectedEmb(e.target.value)}
                style={s.input}
              >
                {EMBEDDINGS.map((opt) => (
                  <option key={opt} value={opt}>
                    {opt}
                  </option>
                ))}
              </select>

              <div>
                <label style={s.label}>Number of Predicted Edges</label>
                <input
                  type="number"
                  min={1}
                  value={predictK}
                  onChange={(e) => setPredictK(e.target.value)}
                  style={s.number}
                />
              </div>

              <div>
                <label style={s.label}>Edge Existence Probability Threshold</label>
                <input
                  type="number"
                  min={0}
                  max={1}
                  step="0.01"
                  value={probThreshold}
                  onChange={(e) => setProbThreshold(e.target.value)}
                  style={s.number}
                />
              </div>

              <Btn onClick={runEmbeddingAndPredict} disabled={embBusy || lpBusy}>
                {embBusy || lpBusy ? "Running…" : "Train & Predict"}
              </Btn>

              {/* <pre style={s.log}>{lpLog}</pre> */}
            </div>
          </section>
        </>
      )}
    </aside>
  );
}
