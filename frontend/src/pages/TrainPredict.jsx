// frontend/src/pages/TrainPredict.jsx
import React, { useState } from "react";
import { api } from "../components/api";

export default function TrainPredict() {
  const [emb, setEmb] = useState("FastRP");
  const [log, setLog] = useState("");
  const [predictK, setPredictK] = useState(100);
  const [negRatio, setNegRatio] = useState(1.0);
  const [outGraph, setOutGraph] = useState("predictedGraph");
  const [threshold, setThreshold] = useState(0.8);

  async function runLP() {
    setLog("Training + Predicting...");
    try {
      const body = {
        embeddingProperty: emb,
        trainGraphName: "trainGraph",
        valGraphName: "valGraph",
        testGraphName: "testGraph",
        negativeRatio: Number(negRatio),
        predictK: Number(predictK),
        candidateMultiplier: 20,
        probThreshold: Number(threshold),
        outputGraphName: outGraph,
      };
      const { data } = await api.post("/lp/run", body);
      setLog(JSON.stringify(data, null, 2));
    } catch (e) {
      setLog((e?.response?.data && JSON.stringify(e.response.data)) || String(e));
    }
  }

  return (
    <div style={{ padding: 16, display: "grid", gap: 12 }}>
      <div>
        <label>Embedding property:&nbsp;</label>
        <input value={emb} onChange={e => setEmb(e.target.value)} style={{ width: 320 }} />
      </div>
      <div>
        <label>Predict K:&nbsp;</label>
        <input type="number" value={predictK} onChange={e => setPredictK(e.target.value)} style={{ width: 120 }} />
      </div>
      <div>
        <label>Negative ratio (per positive):&nbsp;</label>
        <input type="number" step="0.1" value={negRatio} onChange={e => setNegRatio(e.target.value)} style={{ width: 120 }} />
      </div>
      <div>
        <label>Probability threshold (≥):&nbsp;</label>
        <input
          type="number"
          min="0"
          max="1"
          step="0.01"
          value={threshold}
          onChange={e => setThreshold(e.target.value)}
          style={{ width: 120 }}
        />
    </div>
      <div>
        <label>Output graph name:&nbsp;</label>
        <input value={outGraph} onChange={e => setOutGraph(e.target.value)} style={{ width: 240 }} />
      </div>

      <div style={{ display: "flex", gap: 8 }}>
        <button onClick={runLP}>Train + Predict (LP)</button>
      </div>

      <pre style={{ background: "#f7f7f7", padding: 12, minHeight: 200 }}>{log}</pre>
    </div>
  );
}
