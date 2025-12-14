// frontend/src/components/LoadDatasetModal.jsx
import React, { useState } from "react";
import { uploadGraph } from "../components/api";

export default function LoadDatasetModal({ open, onClose }) {
  const [edgesFile, setEdgesFile] = useState(null);
  const [featuresFile, setFeaturesFile] = useState(null);
  const [busy, setBusy] = useState(false);
  const [log, setLog] = useState("");
  const [isTemporal, setIsTemporal] = useState(false);
  const [timestampColumn, setTimestampColumn] = useState("");
  const [datasetName, setDatasetName] = useState("");   // NEW

  if (!open) return null;

  async function handleLoad() {
    if (!edgesFile) {
      setLog(
        "Please choose an edges CSV (must have source,target; optional type; extra columns = edge props)."
      );
      return;
    }
    setBusy(true);
    setLog("Loading…");
    try {
      const res = await uploadGraph(edgesFile, featuresFile, {
        isTemporal,
        timestampColumn,
        dataset_name: datasetName || undefined, 
      });
      setLog(JSON.stringify(res, null, 2));
    } catch (e) {
      setLog(`Load failed: ${e?.response?.data?.detail || e.message}`);
    } finally {
      setBusy(false);
    }
  }

  function handleReset() {
    setEdgesFile(null);
    setFeaturesFile(null);
    setLog("");
    setIsTemporal(false);
    setTimestampColumn("");
    setDatasetName("");        
  }

  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(0,0,0,0.4)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        zIndex: 1000,
      }}
    >
      <div
        style={{
          background: "#111827",
          color: "#fff",
          width: 540,
          maxHeight: "90vh",
          borderRadius: 12,
          boxShadow: "0 20px 60px rgba(0,0,0,0.4)",
          display: "flex",
          flexDirection: "column",
        }}
      >
        {/* header */}
        <div
          style={{
            padding: "12px 16px",
            borderBottom: "1px solid #374151",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
          }}
        >
          <strong>Load Dataset</strong>
          <button
            onClick={onClose}
            style={{
              background: "#374151",
              color: "#fff",
              border: "none",
              padding: "4px 8px",
              borderRadius: 6,
              cursor: "pointer",
              fontSize: 14,
              lineHeight: 1,
            }}
            aria-label="Close"
          >
            ✕
          </button>
        </div>

        {/* body (scrollable if content is tall) */}
        <div
          style={{
            padding: 16,
            display: "grid",
            gap: 12,
            overflowY: "auto",
          }}
        >
          <div>
            <div style={{ fontSize: 12, color: "#9CA3AF", marginBottom: 6 }}>
              Edges CSV (required):{" "}
              <code>source,target[,type][,edge_prop1,...]</code>
            </div>
            <input
              type="file"
              accept=".csv"
              onChange={(e) => setEdgesFile(e.target.files?.[0] || null)}
            />
          </div>

          <div>
            <div style={{ fontSize: 12, color: "#9CA3AF", marginBottom: 6 }}>
              Features CSV (optional): <code>id,prop1,prop2,...</code> (all
              columns after <code>id</code> become node properties)
            </div>
            <input
              type="file"
              accept=".csv"
              onChange={(e) => setFeaturesFile(e.target.files?.[0] || null)}
            />
          </div>

          <div>
            {/* Dataset name (optional) */}
            <div>
            <div style={{ fontSize: 12, color: "#9CA3AF", marginBottom: 4 }}>
              Dataset name (optional):
            </div>
            <input
              type="text"
              value={datasetName}
              onChange={(e) => setDatasetName(e.target.value)}
              placeholder="e.g. my-dataset-1"
              className="mt-1 rounded-md border p-2 bg-gray-800 text-white"
            />
          </div>

          </div>

          {/* Temporal graph options */}
          <div className="mt-3 flex flex-col">
            <label className="inline-flex items-center space-x-2">
              <input
                type="checkbox"
                checked={isTemporal}
                onChange={(e) => setIsTemporal(e.target.checked)}
                className="h-4 w-4 accent-blue-500"
              />
              <span>Import as temporal graph</span>
            </label>

            <input
              type="text"
              placeholder="e.g. timestamp"
              disabled={!isTemporal}
              value={timestampColumn}
              onChange={(e) => setTimestampColumn(e.target.value)}
              className={`mt-2 rounded-md border p-2 bg-gray-800 text-white ${
                !isTemporal ? "opacity-50 cursor-not-allowed" : ""
              }`}
            />
          </div>

          {/* footer buttons (Load + Reset) */}
          <div style={{ display: "flex", gap: 8 }}>
            <button
              onClick={handleLoad}
              disabled={busy || !edgesFile}
              style={{
                padding: "8px 12px",
                border: "1px solid #6B7280",
                background: busy ? "#374151" : "#111827",
                color: "#fff",
                borderRadius: 6,
                cursor: busy || !edgesFile ? "default" : "pointer",
              }}
            >
              {busy ? "Loading…" : "Load"}
            </button>
            <button
              onClick={handleReset}
              style={{
                padding: "8px 12px",
                border: "1px solid #6B7280",
                background: "#111827",
                color: "#fff",
                borderRadius: 6,
                cursor: "pointer",
              }}
            >
              Reset
            </button>
          </div>

          <pre
            style={{
              background: "#0b1220",
              padding: 12,
              borderRadius: 8,
              minHeight: 140,
              whiteSpace: "pre-wrap",
              overflowY: "auto",
            }}
          >
            {log}
          </pre>
        </div>
      </div>
    </div>
  );
}
