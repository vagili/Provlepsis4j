// frontend/src/pages/TrainLP.jsx
import React from "react";
import { runLP } from "../components/api";

export default function TrainLP() {
  const [embeddingProp, setEmbeddingProp] = React.useState("fastRP");
  const [trainName, setTrainName] = React.useState("trainGraph");
  const [valName, setValName] = React.useState("");
  const [testName, setTestName] = React.useState("");
  const [negativeRatio, setNegativeRatio] = React.useState(1.0);

  const [predictK, setPredictK] = React.useState(100);
  const [candidateMultiplier, setCandidateMultiplier] = React.useState(20);
  const [outputGraph, setOutputGraph] = React.useState("predictedGraph");

  const [busy, setBusy] = React.useState(false);
  const [error, setError] = React.useState(null);
  const [result, setResult] = React.useState(null);

  async function onRun() {
    try {
      setBusy(true);
      setError(null);
      setResult(null);

      const payload = {
        embeddingProperty: embeddingProp.trim(),
        trainGraphName: (trainName.trim() || "trainGraph"),
        negativeRatio: Number(negativeRatio) || 1.0,
        predictK: Number(predictK) || 100,
        candidateMultiplier: Number(candidateMultiplier) || 20,
        outputGraphName: (outputGraph.trim() || "predictedGraph"),
      };

      const v = valName?.trim();
      if (v && v.toLowerCase() !== "null") payload.valGraphName = v;

      const t = testName?.trim();
      if (t && t.toLowerCase() !== "null") payload.testGraphName = t;


      const data = await runLP(payload);
      setResult(data);
    } catch (e) {
      setError(e?.response?.data?.detail || e?.message || String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="p-4 space-y-4">
      <h1 className="text-xl font-semibold">Link Prediction (Hadamard + Logistic Regression)</h1>

      <div className="grid md:grid-cols-3 gap-3">
        <label className="text-sm">
          <div className="text-gray-600 mb-1">Embedding Property</div>
          <input className="border rounded w-full px-2 py-1"
                 value={embeddingProp}
                 onChange={e => setEmbeddingProp(e.target.value)}
                 placeholder="fastRP" />
        </label>
        <label className="text-sm">
          <div className="text-gray-600 mb-1">Negatives per Positive</div>
          <input className="border rounded w-full px-2 py-1" type="number" step="0.1" min="0.1" max="10"
                 value={negativeRatio}
                 onChange={e => setNegativeRatio(e.target.value)} />
        </label>
      </div>

      <div className="grid md:grid-cols-3 gap-3">
        <label className="text-sm">
          <div className="text-gray-600 mb-1">Train Graph</div>
          <input className="border rounded w-full px-2 py-1"
                 value={trainName}
                 onChange={e => setTrainName(e.target.value)} />
        </label>
        <label className="text-sm">
          <div className="text-gray-600 mb-1">Validation Graph (optional)</div>
          <input className="border rounded w-full px-2 py-1"
                 value={valName}
                 placeholder="valGraph (optional)"
                 onChange={e => setValName(e.target.value)} />
        </label>
        <label className="text-sm">
          <div className="text-gray-600 mb-1">Test Graph (optional)</div>
          <input className="border rounded w-full px-2 py-1"
                 value={testName}
                 placeholder="testGraph (optional)"
                 onChange={e => setTestName(e.target.value)} />
        </label>
      </div>

      <div className="grid md:grid-cols-3 gap-3">
        <label className="text-sm">
          <div className="text-gray-600 mb-1">Top-K to predict</div>
          <input className="border rounded w-full px-2 py-1" type="number" min="1"
                 value={predictK}
                 onChange={e => setPredictK(e.target.value)} />
        </label>
        <label className="text-sm">
          <div className="text-gray-600 mb-1">Candidate Multiplier</div>
          <input className="border rounded w-full px-2 py-1" type="number" min="2" max="200"
                 value={candidateMultiplier}
                 onChange={e => setCandidateMultiplier(e.target.value)} />
        </label>
        <label className="text-sm">
          <div className="text-gray-600 mb-1">Output Graph Name</div>
          <input className="border rounded w-full px-2 py-1"
                 value={outputGraph}
                 onChange={e => setOutputGraph(e.target.value)} />
        </label>
      </div>

      <div className="flex gap-2 items-center">
        <button className="px-3 py-1 rounded border" onClick={onRun} disabled={busy}>
          {busy ? "Running…" : "Run"}
        </button>
        {error && <div className="text-red-600 text-sm">{error}</div>}
      </div>

      {result && (
        <div className="text-sm text-gray-800 space-y-3">
          <div className="font-semibold">Model</div>
          <pre className="bg-gray-50 p-2 rounded border text-xs overflow-auto">
            {JSON.stringify(result.model, null, 2)}
          </pre>

          <div className="font-semibold">Metrics</div>
          <div>Train — pos {result.train?.count_pos}, neg {result.train?.count_neg}, AUC {fmt(result.train?.auc)}, AP {fmt(result.train?.ap)}, Acc {fmt(result.train?.accuracy)}, P {fmt(result.train?.precision)}, R {fmt(result.train?.recall)}, F1 {fmt(result.train?.f1)}</div>
          {result.validation && (
            <div>Val — pos {result.validation?.count_pos}, neg {result.validation?.count_neg}, AUC {fmt(result.validation?.auc)}, AP {fmt(result.validation?.ap)}, Acc {fmt(result.validation?.accuracy)}, P {fmt(result.validation?.precision)}, R {fmt(result.validation?.recall)}, F1 {fmt(result.validation?.f1)}</div>
          )}
          {result.test && (
            <div>Test — pos {result.test?.count_pos}, neg {result.test?.count_neg}, AUC {fmt(result.test?.auc)}, AP {fmt(result.test?.ap)}, Acc {fmt(result.test?.accuracy)}, P {fmt(result.test?.precision)}, R {fmt(result.test?.recall)}, F1 {fmt(result.test?.f1)}</div>
          )}

          <div className="font-semibold mt-3">Predicted Graph</div>
          <div>
            {result.predicted?.graphName
              ? (<span><b>{result.predicted.graphName}</b> — nodes {result.predicted.nodeCount}, rels {result.predicted.relationshipCount}</span>)
              : "—"}
          </div>
        </div>
      )}
    </div>
  );
}

function fmt(x) {
  if (x === null || x === undefined) return "—";
  return typeof x === "number" ? x.toFixed(4) : String(x);
}
