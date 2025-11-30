// frontend/src/pages/Split.jsx
import React from "react";
import { executeSplit } from "../components/api";

export default function SplitPage() {
  const [holdout, setHoldout] = React.useState(0.1);     // TEST fraction
  const [valHoldout, setValHoldout] = React.useState(0); // VALIDATION fraction
  const [trainName, setTrainName] = React.useState("trainGraph");
  const [testName, setTestName] = React.useState("testGraph");
  const [valName, setValName] = React.useState("valGraph");

  const [busy, setBusy] = React.useState(false);
  const [error, setError] = React.useState(null);
  const [result, setResult] = React.useState(null);

  const onSplit = async () => {
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const h = Number(holdout);
      const v = Number(valHoldout);
      if (isNaN(h) || isNaN(v) || h < 0 || h >= 1 || v < 0 || v >= 1) {
        throw new Error("Hold-out values must be numbers in [0.0, 1.0).");
        }
      if (h + v >= 1.0) {
        throw new Error("testHoldout + valHoldout must be < 1.0.");
      }

      const data = await executeSplit({
        trainGraphName: trainName || "trainGraph",
        testGraphName: testName || "testGraph",
        valGraphName: valName || "valGraph",
        testHoldout: h,
        valHoldout: v,
      });
      setResult(data);

      // Keep your existing alert
      alert(
        `Split complete.\n` +
        `Train: ${data?.train?.relationshipCount ?? 0} rels\n` +
        `Test:  ${data?.test?.relationshipCount ?? 0} rels\n` +
        `Val:   ${data?.validation?.relationshipCount ?? 0} rels`
      );
    } catch (e) {
      setError(e?.message || String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="p-4 space-y-4">
      <h1 className="text-xl font-semibold">Split Graph (Master)</h1>

      <div className="grid md:grid-cols-3 gap-3">
        <label className="text-sm">
          <div className="text-gray-600 mb-1">Train Graph Name</div>
          <input
            className="border rounded w-full px-2 py-1"
            value={trainName}
            onChange={(e) => setTrainName(e.target.value)}
            placeholder="trainGraph"
          />
        </label>
        <label className="text-sm">
          <div className="text-gray-600 mb-1">Test Graph Name</div>
          <input
            className="border rounded w-full px-2 py-1"
            value={testName}
            onChange={(e) => setTestName(e.target.value)}
            placeholder="testGraph"
          />
        </label>
        <label className="text-sm">
          <div className="text-gray-600 mb-1">Validation Graph Name (optional)</div>
          <input
            className="border rounded w-full px-2 py-1"
            value={valName}
            onChange={(e) => setValName(e.target.value)}
            placeholder="valGraph"
          />
        </label>
      </div>

      <div className="grid md:grid-cols-3 gap-3">
        <label className="text-sm">
          <div className="text-gray-600 mb-1">Test Hold-out (fraction 0.0 - 1.0)</div>
          <input
            className="border rounded w-full px-2 py-1"
            type="number"
            step="0.01"
            min="0"
            max="0.99"
            value={holdout}
            onChange={(e) => setHoldout(e.target.value)}
            placeholder="0.10"
          />
        </label>
        <label className="text-sm">
          <div className="text-gray-600 mb-1">Validation Hold-out (fraction 0.0 - 1.0)</div>
          <input
            className="border rounded w-full px-2 py-1"
            type="number"
            step="0.01"
            min="0"
            max="0.99"
            value={valHoldout}
            onChange={(e) => setValHoldout(e.target.value)}
            placeholder="0.00"
          />
        </label>
      </div>

      <div className="flex gap-2">
        <button className="px-3 py-1 rounded border" onClick={onSplit} disabled={busy}>
          {busy ? "Splitting..." : "Split"}
        </button>
        {error && <div className="text-red-600 text-sm">{error}</div>}
      </div>

      {result && (
        <div className="text-sm text-gray-800 space-y-1">
          {/* === Your requested block integrated === */}
          <div>
            <b>Train:</b> {result.train?.graphName ?? "—"} — nodes {result.train?.nodeCount ?? "—"}, rels {result.train?.relationshipCount ?? "—"}
            {typeof result.trainConnected === "boolean" && (
              <> · connected: {result.trainConnected ? "true" : "false"}</>
            )}
            {typeof result.trainComponents === "number" && (
              <> · components: {result.trainComponents}</>
            )}
          </div>

          <div>
            <b>Test:</b> {result.test?.graphName ?? "—"} — nodes {result.test?.nodeCount ?? "—"}, rels {result.test?.relationshipCount ?? "—"}
          </div>

          <div>
            <b>Validation:</b> {result.validation?.graphName ?? "—"} — nodes {result.validation?.nodeCount ?? "—"}, rels {result.validation?.relationshipCount ?? "—"}
          </div>
        </div>
      )}
    </div>
  );
}
